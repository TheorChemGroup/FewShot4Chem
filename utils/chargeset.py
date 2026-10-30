"""Chargeset dataset with gasteiger charges for each atom."""

import logging
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data, OnDiskDataset, download_google_url, extract_zip
from tqdm.auto import tqdm

from utils.constants import ATOM_2_IDX

logger = logging.getLogger(__name__)


class Chargeset(OnDiskDataset):
    """Chargeset dataset with gasteiger charges for each atom."""

    BUFFER_SIZE = 10000
    single_atom = True

    def download(self) -> None:
        """Download the dataset.

        Downloads the dataset from Google Drive and extracts it to the `raw_dir`.

        The dataset is downloaded from the following URL:
        https://drive.google.com/file/d/1AJGk-O99YHe1W2BeH7Q1wJr1mOK4Xkrw/view

        The dataset is extracted from the following file:
        smiles.zip
        """
        download_google_url(
            "1AJGk-O99YHe1W2BeH7Q1wJr1mOK4Xkrw",
            self.raw_dir,
            "smiles.zip",
        )
        extract_zip(str((Path(self.raw_dir) / "smiles.zip").absolute().resolve()), self.raw_dir)

    def raw_file_names(self) -> list[str]:
        """Returns the raw file names."""
        return ["smiles.txt"]

    def processed_file_names(self) -> list[str]:
        """Returns the processed file names."""
        return ["sqlite.db"]

    def process(self) -> None:
        """Process the raw data and generate the Chargeset dataset.

        This method processes the raw data into a Chargeset dataset and saves it to disk.

        The dataset is processed in the following steps:

        1. Reads the raw data from the `raw_dir`.
        2. For each molecule in the raw data:
            a. Computes the atom and edge features.
            b. Computes the charges of the molecule.
            c. Creates a `Data` object from the features and charges.
            d. Adds the `Data` object to a buffer.
        3. Saves the buffer to disk in chunks of `BUFFER_SIZE` elements.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        with Path(self.raw_paths[0]).open(encoding="utf-8") as f:
            smiles_list = f.readlines()

        subset = []

        if logger.isEnabledFor(logging.INFO):
            progress_bar = tqdm(smiles_list, desc="Processing", unit="molecule")
        else:
            progress_bar = smiles_list

        for idx, smiles in enumerate(progress_bar):
            try:
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol is None:
                    continue
                Chem.SanitizeMol(mol)
                Chem.AddHs(mol)
                ComputeGasteigerCharges(mol)
                adj_matrix = Chem.GetAdjacencyMatrix(mol)

                atom_idxs = [ATOM_2_IDX[atom.GetSymbol()] for atom in mol.GetAtoms()]
                atom_idxs = torch.tensor(atom_idxs, dtype=torch.long)

                edge_idxs = torch.tensor(adj_matrix.nonzero(), dtype=torch.long).t().contiguous()
                charges = torch.tensor(
                    [float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()],
                    dtype=torch.float,
                )
                subset.append(
                    Data(
                        x=atom_idxs,
                        edge_index=edge_idxs.T,
                        smiles=smiles,
                        idx=idx,
                        y=charges,
                    ),
                )
            except KeyError:
                pass
            if len(subset) >= self.BUFFER_SIZE or idx == len(smiles_list) - 1:
                self.append(subset)
                subset = []
        logger.info("Processing done")
