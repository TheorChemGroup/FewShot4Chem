"""Provides a dataset filter for filtered datasets creation from the original datasets."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch_geometric
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, download_google_url, extract_zip
from tqdm.auto import tqdm

from utils._no_filtered_datasets import (
    DATASET_MAPPING as NO_FILTERED_DATASET_MAPPING,
)
from utils._no_filtered_datasets import (
    PRE_FILTER_MAPPING as NO_FILTERED_PRE_FILTER_MAPPING,
)
from utils._no_filtered_datasets import (
    TRANSFORM_MAPPING as NO_FILTERED_TRANSFORM_MAPPING,
)


def _no_filtered_load_dataset(
    dataset_name: str,
    root: str | Path = Path("datasets"),
) -> torch_geometric.data.Dataset:
    if isinstance(root, str):
        root = Path(root)
    dataset_class = NO_FILTERED_DATASET_MAPPING[dataset_name]
    transform = NO_FILTERED_TRANSFORM_MAPPING[dataset_name]
    pre_filter = NO_FILTERED_PRE_FILTER_MAPPING[dataset_name]
    return dataset_class(
        root=root / dataset_name,
        transform=transform,
        pre_filter=pre_filter,
    )


logger = logging.getLogger(__name__)

NO_BOND = 0
SINGLE_BOND = 1
DOUBLE_BOND = 2


def MolFromAtomicNumbersAndAdjacencyMatrix(  # noqa: N802 (rdkit-like function may be named not in snake_case)
    atomic_numbers: Sequence[int],
    adjacency_matrix: Sequence[Sequence[int]],
) -> Chem.Mol:
    """Create a molecule from a list of atomic numbers and an adjacency matrix.

    Parameters
    ----------
    atomic_numbers : Sequence[int]
        List of atomic numbers of the atoms in the molecule
    adjacency_matrix : Sequence[Sequence[int]]
        Adjacency matrix of the molecule.
        adjacency_matrix[i][j] contains the bond type between atoms i and j.
        Valid bond types are:
            0: no bond
            1: single bond
            2: double bond

    Returns
    -------
    Chem.Mol
        The constructed molecule

    Notes
    -----
    The function does not check if the provided adjacency matrix is symmetric.
    The function does not remove stereochemistry from the molecule.

    """
    mol = Chem.RWMol()

    node_to_idx = {}
    for i, atomic_number in enumerate(atomic_numbers):
        atom = Chem.Atom(atomic_number)
        molIdx = mol.AddAtom(atom)  # noqa: N806 I'm rdkitten, i have rdpaws
        node_to_idx[i] = molIdx

    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            if iy <= ix:
                continue

            if bond == NO_BOND:
                continue
            if bond == SINGLE_BOND:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == DOUBLE_BOND:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    return mol.GetMol()


class UnknownBondTypeError(ValueError):
    """Unknown bond type error."""


def MolFromAtomicNumbersAndEdges(  # noqa: N802 (rdkit-like function may be named not in snake_case)
    atomic_numbers: Sequence[int],
    edge_index: Sequence[tuple[int, int]],
    edge_attr: Sequence[Any],
) -> Chem.Mol:
    """Construct a molecule from the given atomic numbers, edge indices and edge attributes.

    Parameters
    ----------
    atomic_numbers : Sequence[int]
        The atomic numbers of the atoms in the molecule.
    edge_index : Sequence[tuple[int, int]]
        The indices of connected atoms.
    edge_attr : Sequence[Any]
        The attributes of the edges, that is the bond type.

    Returns
    -------
    The constructed molecule.

    Notes
    -----
    The function does not check if the provided adjacency matrix is symmetric.
    The function does not remove stereochemistry from the molecule.

    """
    mol = Chem.RWMol()

    node_to_idx = {}
    for i, atomic_number in enumerate(atomic_numbers):
        atom = Chem.Atom(atomic_number)
        molIdx = mol.AddAtom(atom)  # noqa: N806 I'm rdkitten, i have rdpaws
        node_to_idx[i] = molIdx

    for (start, end), bond_desc in zip(edge_index, edge_attr, strict=False):
        if bond_desc == NO_BOND:
            continue
        if bond_desc == SINGLE_BOND:
            bond_type = Chem.rdchem.BondType.SINGLE
        elif bond_desc == DOUBLE_BOND:
            bond_type = Chem.rdchem.BondType.DOUBLE
        else:
            raise UnknownBondTypeError(bond_desc)
        mol.AddBond(node_to_idx[start], node_to_idx[end], bond_type)

    return mol.GetMol()


class AtomicNumbersNotFoundError(ValueError):
    """Atomic numbers not found error."""


class MoleculeCreationError(RuntimeError):
    """Molecule creation error."""


class FilteredDataset(InMemoryDataset):
    """Dataste with filtered out molecules from pubchem77M train set."""

    parent_dataset_name: str

    def __init_subclass__(cls, parent_dataset_name: str) -> None:
        """Initialize the subclass of FilteredDataset.

        Is used as a class attribute in the `__init__` method of the subclass to
        set the `parent_dataset` attribute of the instance.

        Parameters
        ----------
        parent_dataset_name : str
            Name of the parent dataset.

        """
        cls.parent_dataset_name = parent_dataset_name
        super().__init_subclass__()

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Initialize a FilteredDataset object.

        Loads the dataset from the processed path.

        Parameters
        ----------
        *args : list[Any]
            Positional arguments to be passed to the parent class.
        **kwargs : dict[str, Any]
            Keyword arguments to be passed to the parent class.

        """
        self.parent_dataset = None
        super().__init__(*args, **kwargs)
        self.load(self.processed_paths[0])

    def raw_file_names(self) -> list[str]:
        """Returns the names of the raw data files.

        The dataset is downloaded from the following URL:
        https://drive.google.com/file/d/1LTkqPvx2bsYbxKyE3C3dsJje4YRn-K2F/view

        The dataset is extracted from the following file:
        pubchem77M.zip

        The files are located in the `raw_dir` directory.

        Returns
        -------
        list[str]
            List of names of the raw data files.

        """
        return ["pubchem77M.txt"]

    def processed_file_names(self) -> list[str]:
        """Returns the names of the processed data files.

        The processed data files are saved in the `processed_dir` directory.

        Returns
        -------
        list[str]
            List of names of the processed data files.

        """
        return ["data.pt"]

    def download(self) -> None:
        """Download the dataset.

        Downloads the dataset from Google Drive and extracts it to the `raw_dir`.

        The dataset is downloaded from the following URL:
        https://drive.google.com/file/d/1LTkqPvx2bsYbxKyE3C3dsJje4YRn-K2F/view

        The dataset is extracted from the following file:
        pubchem77M.zip

        The dataset is extracted to the following directory:
        `raw_dir`

        The dataset is extracted with the following files:
        pubchem77M.txt

        """
        raw_dir_path = Path(self.raw_dir)
        raw_dir_path.mkdir(exist_ok=True, parents=True)
        download_google_url("1LTkqPvx2bsYbxKyE3C3dsJje4YRn-K2F", self.raw_dir, "pubchem77M.zip")
        extract_zip(raw_dir_path / "pubchem77M.zip", self.raw_dir)

    def process(self) -> None:  # IMPROVE: break into smaller functions
        """Processes the dataset.

        The dataset is processed in the following steps:

        1. Loads the taboo smiles from the "taboo.json" file in the `raw_dir`.
        2. If the "taboo.json" file does not exist, it is generated from the
           "pubchem77M.txt" file in the `raw_dir`.
        3. Loads the parent dataset from the `root` directory.
        4. Filters the parent dataset by checking if the smiles is in the taboo
           set. If it is, the data point is skipped.
        5. Saves the filtered dataset to the `processed_paths[0]` file.

        The dataset is saved in a PyTorch Geometric `Data` format.

        The dataset is saved with the following keys:

        - `smiles`: The smiles of the molecule.
        - `atomic_numbers`: The atomic numbers of the atoms in the molecule.
        - `adjacency_matrix`: The adjacency matrix of the molecule.
        - `edge_index`: The edge index of the molecule.
        - `edge_attr`: The edge attributes of the molecule.

        The dataset is saved with the following shape:

        - `smiles`: `(num_samples,)`
        - `atomic_numbers`: `(num_samples, num_atoms)`
        - `adjacency_matrix`: `(num_samples, num_atoms, num_atoms)`
        - `edge_index`: `(num_samples, num_edges, 2)`
        - `edge_attr`: `(num_samples, num_edges)`

        The dataset is saved with the following dtypes:

        - `smiles`: `str`
        - `atomic_numbers`: `torch.long`
        - `adjacency_matrix`: `torch.long`
        - `edge_index`: `torch.long`
        - `edge_attr`: `torch.long`

        Raises
        ------
        AtomicNumbersNotFoundError
            If the atomic numbers cannot be found in the provided molecule info
        MoleculeCreationError
            If the molecule cannot be created from the provided molecule info

        Returns
        -------
        None

        """
        raw_dir_path = Path(self.raw_dir)
        try:
            with (raw_dir_path / "taboo.json").open(encoding="utf-8") as f:
                logger.debug(
                    "Loading taboo smiles from file %s",
                    raw_dir_path / "taboo.json",
                )
                taboo = set(json.load(f))
                logger.info(
                    "Taboo smiles successfully found at %s",
                    raw_dir_path / "taboo.json",
                )
        except (FileNotFoundError, json.JSONDecodeError):
            with Path(self.raw_paths[0]).open(encoding="utf-8") as f:
                logger.info("Loading taboo smiles...")
                if logger.isEnabledFor(logging.INFO):
                    progress_bar = tqdm(f.readlines(), desc="Loading taboo smiles")
                else:
                    progress_bar = f.readlines()
                taboo = set()
                for smiles in progress_bar:
                    mol = Chem.MolFromSmiles(smiles.strip())
                    if mol is not None:
                        status = Chem.SanitizeMol(mol, catchErrors=True)
                        if status == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                            Chem.RemoveStereochemistry(mol)
                            taboo.add(Chem.MolToSmiles(mol))

            with (raw_dir_path / "taboo.json").open("w", encoding="utf-8") as f:
                json.dump(list(taboo), f)

        parent_dataset = _no_filtered_load_dataset(
            self.parent_dataset_name,
            root=Path(self.root).parent,
        )

        if logger.isEnabledFor(logging.INFO):
            progress_bar = tqdm(parent_dataset, desc="Filtering...")
        else:
            progress_bar = parent_dataset

        result_data = []

        for i, data in enumerate(progress_bar):
            if (i % 100000 == 0 or i == len(parent_dataset) - 1) and logger.isEnabledFor(
                logging.INFO,
            ):
                progress_bar.set_postfix(processed=len(result_data))
            try:
                mol = Chem.MolFromSmiles(data.smiles)
            except AttributeError as smiles_not_found:
                mol = None
                atomic_numbers_candidates = [
                    "atomic_numbers",
                    "species_periodic_table_index",
                    "z",
                    "x",
                ]
                for atomic_numbers_candidate in atomic_numbers_candidates:
                    try:
                        atomic_numbers = data[atomic_numbers_candidate]
                        break
                    except (TypeError, AttributeError):
                        continue
                else:
                    raise AtomicNumbersNotFoundError(data) from smiles_not_found
                try:
                    adj_matrix = data["adjacency_matrix"]
                    mol = MolFromAtomicNumbersAndAdjacencyMatrix(atomic_numbers, adj_matrix)
                except (TypeError, AttributeError):
                    try:
                        edge_index = data["edge_index"]
                        edge_attr = data["edge_attr"]
                        mol = MolFromAtomicNumbersAndEdges(atomic_numbers, edge_index, edge_attr)
                    except (TypeError, AttributeError):
                        raise MoleculeCreationError(data) from smiles_not_found

            if mol is None:
                continue
            status = Chem.SanitizeMol(mol, catchErrors=True)
            if status != Chem.SanitizeFlags.SANITIZE_NONE or mol is None:
                continue
            Chem.RemoveStereochemistry(mol)
            smiles = Chem.MolToSmiles(mol)

            if smiles not in taboo:
                data.smiles = smiles
                result_data.append(data)

        self.save(result_data, self.processed_paths[0])
