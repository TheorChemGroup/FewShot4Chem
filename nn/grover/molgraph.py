"""Molecular graph data structures for GROVER.

Adapted from https://github.com/tencent-ailab/grover
"""

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem

MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
    """Gets the dimensionality of atom features.

    Returns
    -------
    int
        Atom feature dimension.

    """
    return ATOM_FDIM + 18


def get_bond_fdim() -> int:
    """Gets the dimensionality of bond features.

    Returns
    -------
    int
        Bond feature dimension.

    """
    return BOND_FDIM


def onek_encoding_unk(value: int, choices: list[int]) -> list[int]:
    """Creates a one-hot encoding.

    Parameters
    ----------
    value : int
        The value for which the encoding should be one.
    choices : list[int]
        A list of possible values.

    Returns
    -------
    list[int]
        A one-hot encoding of the value.

    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


@dataclass
class GroverArgs:
    """Arguments for GROVER model configuration.

    Attributes
    ----------
    hidden_size : int
        Hidden size of the model.
    dropout : float
        Dropout rate.
    activation : str
        Activation function name.
    depth : int
        Message passing depth.
    num_mt_block : int
        Number of multi-head attention blocks.
    num_attn_head : int
        Number of attention heads.
    bias : bool
        Whether to use bias in linear layers.
    undirected : bool
        Whether to use undirected message passing.
    dense : bool
        Whether to use dense connections.
    cuda : bool
        Whether to use CUDA.
    bond_drop_rate : float
        Rate of dropping bonds during training.
    no_cache : bool
        Whether to disable caching.

    """

    hidden_size: int = 128
    dropout: float = 0.1
    activation: str = "PReLU"
    depth: int = 3
    num_mt_block: int = 1
    num_attn_head: int = 4
    bias: bool = False
    undirected: bool = True
    dense: bool = True
    cuda: bool = False
    bond_drop_rate: float = 0.0
    no_cache: bool = True
    backbone: str = "gtrans"
    embedding_output_type: str = "both"
    fingerprint_source: str = "both"


class MolGraph:
    """A MolGraph represents the graph structure of a single molecule.

    Attributes
    ----------
    smiles : str
        SMILES string.
    n_atoms : int
        The number of atoms in the molecule.
    n_bonds : int
        The number of bonds in the molecule.
    f_atoms : list[list[float]]
        Atom features for each atom.
    f_bonds : list[list[float]]
        Bond features for each bond.
    a2b : list[list[int]]
        Mapping from atom index to incoming bond indices.
    b2a : list[int]
        Mapping from bond index to the atom the bond originates from.
    b2revb : list[int]
        Mapping from bond index to the reverse bond index.

    """

    def __init__(self, smiles: str, args: GroverArgs | None = None) -> None:
        """Computes the graph structure and featurization of a molecule.

        Parameters
        ----------
        smiles : str
            A SMILES string.
        args : GroverArgs | None, optional
            Arguments for graph construction.

        """
        if args is None:
            args = GroverArgs()

        self.smiles = smiles
        self.args = args
        self.n_atoms = 0
        self.n_bonds = 0
        self.f_atoms: list[list[float]] = []
        self.f_bonds: list[list[float]] = []
        self.a2b: list[list[int]] = []
        self.b2a: list[int] = []
        self.b2revb: list[int] = []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
            "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
        )
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),"
            "$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),"
            "$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]",
        )

        self.hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        self.hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        self.acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        self.basic_match = sum(mol.GetSubstructMatches(self.basic), ())
        self.ring_info = mol.GetRingInfo()

        self.n_atoms = mol.GetNumAtoms()

        for atom in mol.GetAtoms():
            self.f_atoms.append(self._atom_features(atom))
        self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

        for _ in range(self.n_atoms):
            self.a2b.append([])

        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                if args.bond_drop_rate > 0:
                    if np.random.binomial(1, args.bond_drop_rate):
                        continue

                f_bond = self._bond_features(bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

    def _atom_features(self, atom: Chem.rdchem.Atom) -> list[float]:
        """Builds a feature vector for an atom.

        Parameters
        ----------
        atom : Chem.rdchem.Atom
            An RDKit atom.

        Returns
        -------
        list[float]
            A list containing the atom features.

        """
        features = (
            onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
            + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
            + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
            + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
            + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
            + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
            + [1 if atom.GetIsAromatic() else 0]
            + [atom.GetMass() * 0.01]
        )
        atom_idx = atom.GetIdx()
        features = (
            features
            + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
            + [atom_idx in self.hydrogen_acceptor_match]
            + [atom_idx in self.hydrogen_donor_match]
            + [atom_idx in self.acidic_match]
            + [atom_idx in self.basic_match]
            + [
                self.ring_info.IsAtomInRingOfSize(atom_idx, 3),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 4),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 5),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 6),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 7),
                self.ring_info.IsAtomInRingOfSize(atom_idx, 8),
            ]
        )
        return features

    def _bond_features(self, bond: Chem.rdchem.Bond) -> list[float]:
        """Builds a feature vector for a bond.

        Parameters
        ----------
        bond : Chem.Bond
            An RDKit bond.

        Returns
        -------
        list[float]
            A list containing the bond features.

        """
        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated() if bt is not None else 0,
                bond.IsInRing() if bt is not None else 0,
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond


class BatchMolGraph:
    """A BatchMolGraph represents a batch of molecular graphs.

    Attributes
    ----------
    smiles_batch : list[str]
        List of SMILES strings.
    n_mols : int
        Number of molecules in the batch.
    atom_fdim : int
        Dimensionality of atom features.
    bond_fdim : int
        Dimensionality of bond features.
    a_scope : list[tuple[int, int]]
        List of (start, num_atoms) for each molecule.
    b_scope : list[tuple[int, int]]
        List of (start, num_bonds) for each molecule.
    max_num_bonds : int
        Maximum number of bonds neighboring an atom.

    """

    def __init__(self, mol_graphs: list[MolGraph], args: GroverArgs | None = None) -> None:
        """Initialize a BatchMolGraph from a list of MolGraphs.

        Parameters
        ----------
        mol_graphs : list[MolGraph]
            List of molecular graphs.
        args : GroverArgs | None, optional
            Arguments for batch construction.

        """
        if args is None:
            args = GroverArgs()

        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + self.atom_fdim

        self.n_atoms = 1
        self.n_bonds = 1
        self.a_scope: list[tuple[int, int]] = []
        self.b_scope: list[tuple[int, int]] = []

        f_atoms: list[list[float]] = [[0.0] * self.atom_fdim]
        f_bonds: list[list[float]] = [[0.0] * self.bond_fdim]
        a2b: list[list[int]] = [[]]
        b2a: list[int] = [0]
        b2revb: list[int] = [0]

        for mol_graph in mol_graphs:
            if mol_graph.n_atoms == 0:
                continue

            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) if a2b else 1

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor(
            [a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)],
        )
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None
        self.a2a = self.b2a[self.a2b]
        self.a_scope_tensor = torch.LongTensor(self.a_scope)
        self.b_scope_tensor = torch.LongTensor(self.b_scope)

    def get_components(
        self,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
        torch.LongTensor,
        list[tuple[int, int]],
        list[tuple[int, int]],
        torch.LongTensor,
    ]:
        """Returns the components of the BatchMolGraph.

        Returns
        -------
        tuple
            Atom features, bond features, a2b, b2a, b2revb, a_scope, b_scope, a2a.

        """
        return (
            self.f_atoms,
            self.f_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
            self.a2a,
        )

    def get_b2b(self) -> torch.LongTensor:
        """Computes mapping from each bond to incoming bonds.

        Returns
        -------
        torch.LongTensor
            Mapping from bond index to incoming bond indices.

        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()
            self.b2b = b2b * revmask
        return self.b2b


def mol2graph(smiles_batch: list[str], args: GroverArgs | None = None) -> BatchMolGraph:
    """Converts a list of SMILES strings to a BatchMolGraph.

    Parameters
    ----------
    smiles_batch : list[str]
        A list of SMILES strings.
    args : GroverArgs | None, optional
        Arguments for graph construction.

    Returns
    -------
    BatchMolGraph
        Batch of molecular graphs.

    """
    if args is None:
        args = GroverArgs()

    mol_graphs = []
    for smiles in smiles_batch:
        mol_graph = MolGraph(smiles, args)
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)
