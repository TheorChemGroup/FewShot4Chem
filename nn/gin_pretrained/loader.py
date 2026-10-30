"""Atom/bond feature vocabulary tables + SMILES → PyG Data converter.

Verbatim port from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
to ensure feature indices match Hu et al.'s pretrained checkpoints.
"""

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def mol_to_graph_data_obj_simple(mol: Chem.Mol) -> Data:
    """Convert RDKit mol → PyG Data with [atomic_num_index, chirality_index] atom features."""
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum()),
            allowable_features["possible_chirality_list"].index(atom.GetChiralTag()),
        ]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features["possible_bonds"].index(bond.GetBondType()),
                allowable_features["possible_bond_dirs"].index(bond.GetBondDir()),
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
