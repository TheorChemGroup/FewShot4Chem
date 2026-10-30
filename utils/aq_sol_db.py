"""AqSolDB dataset."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from utils.constants import ATOM_2_IDX


class AqSolDB(InMemoryDataset):
    """AqSolDB dataset.

    AqSolDB is a dataset of molecules with their measured aqueous solubility values.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    transform : Optional[Callable[[Data], Data]], optional
        A function/transform that takes in a :class:`Data` object and returns a
        transformed version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform : Optional[Callable[[Data], Data]], optional
        A function/transform that takes in a :class:`Data` object and returns a
        transformed version. The data object will be transformed before being saved to disk.
        (default: :obj:`None`)
    pre_filter : Optional[Callable[[Data], bool]], optional
        A function that takes in a :class:`Data` object and returns a boolean value,
        indicating whether the data object should be included in the final dataset.
        (default: :obj:`None`)

    Attributes
    ----------
    raw_file_names : list[str]
        File names of the raw data.
    processed_file_names : list[str]
        File names of the processed data.
    num_classes : int
        Number of classes in the dataset.

    """

    def __init__(
        self,
        root: str,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> None:
        """Initializes an AqSolDB dataset.

        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        transform : Optional[Callable[[Data], Data]], optional
            A function/transform that takes in a :class:`Data` object and returns a
            transformed version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform : Optional[Callable[[Data], Data]], optional
            A function/transform that takes in a :class:`Data` object and returns a
            transformed version. The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter : Optional[Callable[[Data], bool]], optional
            A function that takes in a :class:`Data` object and returns a boolean value,
            indicating whether the data object should be included in the final dataset.
            (default: :obj:`None`)

        """
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def raw_file_names(self) -> list[str]:
        """Returns the file names of the raw data."""
        return ["curated-solubility-dataset.csv"]

    def processed_file_names(self) -> list[str]:
        """Returns the file names of the processed data."""
        return ["data.pt"]

    def download(self) -> None:
        """Downloads the AqSolDB dataset.

        Downloads the dataset from the following URL:
        https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8#.

        Extracts the dataset from the `curated-solubility-dataset.zip` file
        to the `raw_dir` directory.
        """
        download_url(
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8#",
            self.raw_dir,
            "curated-solubility-dataset.zip",
        )
        extract_zip(
            (Path(self.raw_dir) / "curated-solubility-dataset.zip").absolute().resolve(),
            self.raw_dir,
        )

    def process(self) -> None:
        """Process the raw data and save it to disk.

        The raw data is processed in the following steps:

        1. Reads the raw data from the `raw_dir`.
        2. For each molecule in the raw data:
            a. Computes the atom and edge features.
            b. Creates a `Data` object from the features.
            c. Adds the `Data` object to a buffer.
        3. Saves the buffer to disk in the `processed_paths[0]` file.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        desired_keys = [
            "Solubility",
            "MolLogP",
            "MolMR",
            "TPSA",
            "LabuteASA",
            "BertzCT",
        ]

        raw_data = pd.read_csv(
            self.raw_paths[0],
        )
        processed_data_list = []
        for _, row in raw_data.iterrows():
            try:
                mol = Chem.MolFromSmiles(row["SMILES"])
                if mol is None:
                    continue
                Chem.SanitizeMol(mol)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                if mol is None:
                    continue
                adj_matrix = Chem.GetAdjacencyMatrix(mol)
                atom_idxs = [ATOM_2_IDX[atom.GetSymbol()] for atom in mol.GetAtoms()]
                atom_idxs = torch.tensor(atom_idxs, dtype=torch.long)
                edge_idxs = torch.tensor(adj_matrix.nonzero(), dtype=torch.long).t().contiguous()
                y_aslist = [0] * len(desired_keys)
                for j, key in enumerate(desired_keys):
                    y_aslist[j] = row[key]
                y = torch.tensor(y_aslist, dtype=torch.float)
                smiles_large = Chem.MolToSmiles(mol)
                mol = Chem.MolFromSmiles(smiles_large)
                Chem.RemoveStereochemistry(mol)
                smiles = Chem.MolToSmiles(mol)
                data = Data(
                    smiles=smiles,
                    x=atom_idxs,
                    edge_index=edge_idxs.T,
                    y=y,
                )
                processed_data_list.append(data)
            except KeyError:
                continue
        self.save(processed_data_list, self.processed_paths[0])
