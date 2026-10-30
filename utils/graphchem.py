"""Provides PubChem77M dataset in a graph format."""

import json
import logging
import os
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import torch
import torch_geometric
import yaml
from omegaconf import OmegaConf
from rdkit import Chem
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from torch_geometric.data import Data, OnDiskDataset, download_google_url, extract_zip
from tqdm.auto import tqdm
from yaml.error import YAMLError

from utils.constants import ATOM_2_IDX
from utils.load_tokenizer import get_trainer_params, load_tokenizer, load_tokenizer_config
from utils.transforms import (
    AllPreFilters,
    PreFilterCheckValidSMILES,
    PreFilterRemoveLongSMILES,
)

logger = logging.getLogger(__name__)


def bond_features(bond: Chem.Bond) -> torch.Tensor:
    """Create a feature vector for a bond.

    The feature vector is a binary vector of length 5, where the first four
    elements indicate the type of the bond (single, double, triple, aromatic)
    and the fifth element indicates if the bond is in a ring or not.

    Parameters
    ----------
    bond: Chem.Bond
        The bond to create the feature vector for.

    Returns
    -------
    torch.Tensor
        A binary tensor of length 5 representing the bond features.

    """
    bond_type = bond.GetBondType()
    return torch.tensor(
        [
            int(bond_type == Chem.BondType.SINGLE),
            int(bond_type == Chem.BondType.DOUBLE),
            int(bond_type == Chem.BondType.TRIPLE),
            int(bond_type == Chem.BondType.AROMATIC),
            int(bond.IsInRing()),
        ],
        dtype=torch.float,
    )


def safe_copyattr(obj: object, attr: str, dst: object) -> None:
    """Safely copy an attribute from one object to another.

    If the source object does not have the specified attribute, this function
    does nothing.

    Parameters
    ----------
    obj: object
        The source object.
    attr: str
        The name of the attribute to copy.
    dst: object
        The destination object.

    """
    if hasattr(obj, attr):
        setattr(dst, attr, getattr(obj, attr))


class SplitRememberingDataset:
    """A dataset that remembers its split into training, validation and test sets.

    When calling the `split` method, the dataset is split into training,
    validation and test sets and the indices of each set are saved to disk.
    When calling the `split` method again, the previously saved split is
    loaded from disk.

    Attributes
    ----------
    split_path: ClassVar[Path]
        The path to the file containing the split information.

    """

    def split(
        self,
    ) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
        """Split the dataset into training, validation and test sets.

        If the dataset has already been split and the split is saved to disk,
        load the split from disk. Otherwise, split the dataset randomly into
        training, validation and test sets with a ratio of 0.8, 0.1 and 0.1
        respectively. Save the split to disk.

        Returns
        -------
        tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]
            A tuple of three datasets for training, validation and testing
            respectively.

        """
        splits = self.get_splits()
        if splits:
            logger.debug("Making split subsets from saved splits")
            res = [
                torch.utils.data.Subset(self, splits[split_name])
                for split_name in ["train", "val", "test"]
            ]
            for item in res:
                safe_copyattr(self, "collate_fn", item)
                safe_copyattr(self, "tokenizer", item)
            return res
        train, val, test = torch.utils.data.random_split(self, [0.8, 0.1, 0.1])
        raw_dir = Path(self.raw_dir)
        with (raw_dir / "train.json").open("w", encoding="utf-8") as f:
            logger.debug("Saving train split to %s", raw_dir / "train.json")
            json.dump(train.indices, f)
        with (raw_dir / "val.json").open("w", encoding="utf-8") as f:
            logger.debug("Saving val split to %s", raw_dir / "val.json")
            json.dump(val.indices, f)
        with (raw_dir / "test.json").open("w", encoding="utf-8") as f:
            logger.debug("Saving test split to %s", raw_dir / "test.json")
            json.dump(test.indices, f)
        return train, val, test

    def get_splits(self) -> dict:
        """Returns a dictionary of train, validation and test splits.

        The dictionary contains the following keys:

        - `train`: A list of indices of the training set.
        - `val`: A list of indices of the validation set.
        - `test`: A list of indices of the test set.

        Returns
        -------
        dict
            A dictionary of train, validation and test splits.

        """
        raw_dir = Path(self.raw_dir)
        res = {}
        try:
            for split_name in ["train", "val", "test"]:
                with (raw_dir / (split_name + ".json")).open(encoding="utf-8") as f:
                    logger.debug("Loading split %s", split_name)
                    res[split_name] = json.load(f)
        except FileNotFoundError:
            return {}
        else:
            return res


class GraphChem(OnDiskDataset, SplitRememberingDataset):
    """PubChem77M dataset in a graph format.

    The dataset is a variant of the PubChem77M dataset, which is a subset of
    the PubChem database containing 77 million molecules. The dataset is
    provided in a graph format, where each molecule is represented as a graph
    of atoms connected by bonds. The dataset is stored in a custom format,
    which is a binary file containing a sequence of graphs.

    The dataset is split into training, validation, and test sets. The training
    set contains 70000000 molecules, the validation set contains 7000000
    molecules, and the test set contains 1000000 molecules.

    The dataset provides a variety of features for each molecule, including
    atom and bond features, as well as a variety of molecular properties.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    transform : Optional[Callable], optional
        A function/transform that takes in an object and returns a transformed version.
        The data object will be transformed before every access. By default, None.
    pre_filter : Optional[Callable], optional
        A function/transform that takes in an object and returns a transformed version.
        The data object will be transformed before being saved to disk. By default, None.
    size : Optional[int], optional
        The size of a dataset (No. of samples). None for all 77 million.
        By default, None.
    masked_fractions: dict[str, float]
        A dict with the following keys:
            "nodes": The fraction of nodes to mask. By default 0.15.
            "edges": The fraction of edges to mask. By default 0.05.
    masking_method: str
        The method to use for masking. By default "simple".
    p: Optional[float]
        The probability parameter for recursive masking. By default 0.8.
    buffer_size: int
        The buffer size for the dataset. By default 100000.

    """

    def __init__(  # noqa: PLR0913
        self,
        root: str,
        *,
        transform: Callable | None = None,
        pre_filter: Callable | None = None,
        size: int | None = None,
        masked_fractions: float | None = None,
        masking_method: Literal["simple", "recursive", "synthons"] = "simple",
        p: float = 0.8,
        buffer_size: int = 100000,
    ) -> None:
        """Initialize a GraphChem dataset.

        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        transform : Optional[Callable], optional
            A function/transform that takes in an object and returns a transformed version.
            The data object will be transformed before every access. By default, None.
        pre_filter : Optional[Callable], optional
            A function/transform that takes in an object and returns a transformed version.
            The data object will be transformed before being saved to disk. By default, None.
        size : Optional[int], optional
            The size of a dataset (No. of samples). None for all 77 million.
            By default, None.
        masked_fractions: dict[str, float]
            A dict with the following keys:
                "nodes": The fraction of nodes to mask. By default 0.15.
                "edges": The fraction of edges to mask. By default 0.05.
        masking_method: str
            The method to use for masking. By default "simple".
        p: Optional[float]
            The probability parameter for recursive masking. By default 0.8.
        buffer_size: int
            The buffer size for the dataset. By default 100000.

        """
        logger.debug(
            "GraphChem initialized with root: %s, size: %s, masked_fractions: %s, "
            "masking_method: %s, p: %s, buffer_size: %s, pre_filter: %s, transform: %s",
            root,
            size,
            masked_fractions,
            masking_method,
            p,
            buffer_size,
            pre_filter,
            transform,
        )
        self.BUFFER_SIZE = buffer_size
        self.root = os.path.join(root, "graph")  # noqa: PTH118
        self.masking_method = masking_method
        self.size = size
        self.masked_fractions = (
            masked_fractions if masked_fractions is not None else {"nodes": 0.15, "edges": 0.05}
        )
        self.p = p
        db_file_path = Path(self.processed_paths[0])
        try:
            params_file_path = Path(self.processed_paths[1])
            with params_file_path.open(encoding="utf-8") as f:
                params = yaml.safe_load(f)
                try:
                    if (
                        self.size != params["size"]
                        or self.masked_fractions != params["masked_fractions"]
                    ):
                        logger.warning(
                            "Dataset found at %s appears to have processing parameters"
                            "%s other than passed to __init__("
                            "%s), reprocessing",
                            root,
                            params,
                            self.get_processing_params_as_dict(),
                        )
                        params_file_path.unlink(missing_ok=True)
                        db_file_path.unlink(missing_ok=True)
                except KeyError:
                    logger.warning(
                        "Processing parameters file %s appears to be corrupted, reprocessing",
                        params_file_path,
                    )
                    params_file_path.unlink(missing_ok=True)
                    db_file_path.unlink(missing_ok=True)

        except (FileNotFoundError, YAMLError):
            db_file_path.unlink(missing_ok=True)
        OnDiskDataset.__init__(self, root=root, transform=transform, pre_filter=pre_filter)

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
        return ["sqlite.db", "parameters.yaml"]

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

    def process(self) -> None:
        """Process the raw data and generate the GraphChem dataset.

        This method processes the raw data into a GraphChem dataset and saves it to disk.

        The dataset is processed in the following steps:

        1. Reads the raw data from the `raw_dir`.
        2. For each molecule in the raw data:
            a. Computes the atom and edge features.
            b. Creates a `Data` object from the features.
            c. Adds the `Data` object to a buffer.
        3. Saves the buffer to disk in chunks of `BUFFER_SIZE` elements.
        4. Saves the processing parameters to disk in a file named `parameters.yaml`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        (Path(self.root) / "processed").mkdir(exist_ok=True, parents=True)
        raw_file_path = Path(self.raw_dir) / self.raw_file_names()[0]
        pre_filter = self.pre_filter if self.pre_filter is not None else lambda _: True
        with raw_file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            progress_bar = tqdm(lines) if logger.isEnabledFor(logging.INFO) else lines
            buffer = []
            selected_masking_method = {
                "simple": self._mask_nodes_simple,
                "recursive": self._mask_nodes_recursive,
            }[self.masking_method]
            msk_id_nodes = len(ATOM_2_IDX)
            for idx, smiles_unstripped in enumerate(progress_bar):
                smiles = smiles_unstripped.strip()
                if idx % 10000 == 0 and logger.isEnabledFor(logging.INFO):
                    progress_bar.set_postfix(current=len(self), buffer=len(buffer))
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)

                    atom_idxs = [ATOM_2_IDX.get(atom.GetSymbol()) for atom in mol.GetAtoms()]
                    if None in atom_idxs:
                        continue
                    atom_idxs = torch.tensor(atom_idxs, dtype=torch.long)

                    edge_idxs_set = set()
                    edge_idxs = []
                    edge_attr = []
                    for bond in mol.GetBonds():
                        to_add = (
                            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
                        )
                        for pair in to_add:
                            if pair not in edge_idxs_set:
                                edge_idxs_set.add(pair)
                                edge_idxs.append(pair)
                                edge_attr.append(bond_features(bond))
                    edge_idxs = torch.tensor(edge_idxs, dtype=torch.long).t().contiguous()
                    edge_attr = torch.stack(edge_attr)

                    atom_idxs_unmasked = atom_idxs.clone()
                    atom_idxs = selected_masking_method(
                        atom_idxs,
                        edge_idxs,
                        edge_attr,
                        msk_id_nodes,
                    )

                    new = Data(
                        x=atom_idxs,
                        edge_index=edge_idxs,
                        edge_attr=edge_attr,
                        x_unmasked=atom_idxs_unmasked,
                        smiles=smiles,
                        idx=idx,
                    )
                    if pre_filter(new):
                        buffer.append(new)
                if (
                    len(buffer) >= self.BUFFER_SIZE
                    or idx == len(progress_bar) - 1
                    or (self.size is not None and len(self) + len(buffer) == self.size)
                ):
                    self.extend(buffer)
                    buffer = []
                if self.size is not None and len(self) >= self.size:
                    break
        params_file_path = Path(self.processed_paths[1])
        with params_file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.get_processing_params_as_dict(),
                f,
            )

    def _mask_nodes_simple(
        self,
        atom_idxs: torch.Tensor,
        edge_idxs: torch.Tensor,  # noqa: ARG002
        edge_attr: torch.Tensor,  # noqa: ARG002
        msk_id_nodes: int,
    ) -> torch.Tensor:
        num_nodes = atom_idxs.shape[0]
        num_nodes_to_mask = int(self.masked_fractions["nodes"] * num_nodes)
        idxs_to_mask = torch.randperm(num_nodes)[:num_nodes_to_mask]
        atom_idxs[idxs_to_mask] = msk_id_nodes
        return atom_idxs

    def _mask_nodes_recursive(
        self,
        atom_idxs: torch.Tensor,
        edge_idxs: torch.Tensor,
        edge_attr: torch.Tensor,  # noqa: ARG002
        msk_id_nodes: int,
    ) -> torch.Tensor:
        num_nodes = atom_idxs.shape[0]
        num_nodes_to_mask = max(1, int(self.masked_fractions["nodes"] * num_nodes))
        last_masked_node = None
        unmasked_nodes = set(range(num_nodes))
        for _ in range(num_nodes_to_mask):
            children = [
                v for u, v in edge_idxs.t() if u == last_masked_node and v in unmasked_nodes
            ]
            mask_new_node = last_masked_node is None or not children or (random.random() > self.p)  # noqa: S311 # This ain't cryptography
            if mask_new_node:
                node_to_mask = random.choice(list(unmasked_nodes))  # noqa: S311
            else:
                node_to_mask = random.choice(children)  # noqa: S311
            atom_idxs[node_to_mask] = msk_id_nodes
            unmasked_nodes.remove(node_to_mask)
            new_children = [
                v for u, v in edge_idxs.t() if u == node_to_mask and v in unmasked_nodes
            ]
            if new_children:
                last_masked_node = node_to_mask
        return atom_idxs

    def get_processing_params_as_dict(self) -> dict:
        """Returns a dictionary of processing parameters.

        The dictionary contains the following keys:

        - `size`: The number of data points in the dataset.
        - `masked_fractions`: A dictionary of masked fractions for each of the
          node, edge and graph elements.

        Returns
        -------
        dict
            A dictionary of processing parameters.

        """
        return {
            "size": self.size,
            "masked_fractions": self.masked_fractions,
        }

    def get_dataset_info(self) -> dict:
        """Returns a dictionary containing information about the dataset.

        The dictionary contains the following keys:

        - `processing_params`: A dictionary of processing parameters.
        - `n_samples`: The number of data points in the dataset.

        Returns
        -------
        dict
            A dictionary containing information about the dataset.

        """
        return {
            "processing_params": self.get_processing_params_as_dict(),
            "n_samples": len(self),
        }

    def maybe_save_as_txt(self, format_: str = "SMILES") -> None:
        """Save the dataset to a text file if it doesn't already exist.

        The file is saved in the directory with the same name as the dataset
        but with a `_txt` suffix. The file contains one string per line.

        Parameters
        ----------
        format_ : str, default="SMILES"
            The format of the file. Currently only "SMILES" is supported.

        Returns
        -------
        None

        """
        format_ = format_.lower()
        save_folder = (
            Path(self.root).parent / (Path(self.root).parts[-1] + "_txt") / format_ / "raw"
        )
        save_path = save_folder / "data.txt"
        info_path = save_folder / "info.json"
        if save_path.exists() and info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            if info == self.get_dataset_info():
                logger.info("Dataset already saved to %s", save_path)
                return
        save_folder.mkdir(exist_ok=True, parents=True)
        with save_path.open("w", encoding="utf-8") as f:
            progress_bar = tqdm(self) if logger.isEnabledFor(logging.INFO) else self
            for data in progress_bar:
                f.write(data.smiles + "\n")
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(self.get_dataset_info(), f)
        if self.get_splits():
            raw_dir = Path(self.raw_dir)
            for split_name in ["train", "val", "test"]:
                shutil.copy((raw_dir / f"{split_name}.json"), save_folder / f"{split_name}.json")
        logger.info("Dataset saved to %s", save_path)

    @staticmethod
    def collate_fn(batch: list[Data]) -> torch_geometric.data.Batch:
        """Collate function for the dataset.

        This function takes a list of Data objects and returns a Batch object.

        Parameters
        ----------
        batch : list[Data]
            A list of Data objects to be collated.

        Returns
        -------
        Batch
            A Batch object containing the collated data.

        """
        return torch_geometric.data.Batch.from_data_list(batch)


class GraphChemTxt(torch.utils.data.Dataset, SplitRememberingDataset):
    """A class for loading GraphChem dataset as a collection of tokenized strings from a text file.

    This class assumes that the text file contains one string per line. The
    string is in the format specified by the `format_` parameter. Currently
    only "SMILES" is supported.

    Parameters
    ----------
    root : str
        The root directory of the dataset.
    tokenizer_name : str
        The name of the tokenizer to use.
    external_name : Optional[str], default=None
        The external name of the dataset. If None, it is the same as the root
        directory.
    format_ : str, default="SMILES"
        The format of the text file. Currently only "SMILES" is supported.
    size : Optional[int], default=None
        The number of data points to load. If None, all data points are loaded.
    masked_fractions : Optional[dict[str, float]], default=None
        A dictionary of masked fractions for each atom type. If None, all atoms
        are unmasked.

    Attributes
    ----------
    root : Path
        The root directory of the dataset.
    tokenizer : Tokenizer
        The tokenizer used to tokenize the strings.
    PAD_TOKEN : str
        The token used to pad the strings.

    """

    root: Path
    tokenizer: Tokenizer
    PAD_TOKEN: ClassVar[str] = "[PAD]"  # noqa: S105 This is not a password
    MASK_TOKEN: ClassVar[str] = "[MASK]"  # noqa: S105 This is not a password too

    def __init__(  # noqa:PLR0913
        self,
        root: str | Path,
        tokenizer_name: str,
        external_name: str | None = None,
        format_: str = "SMILES",
        size: int | None = None,
        masked_fractions: dict[str, float] | None = None,
        vocab_size: int | None = None,
        max_smiles_len: int = 300,
    ) -> None:
        """Initialize a GraphChemTxt dataset.

        Parameters
        ----------
        root : str
            The root directory of the dataset.
        tokenizer_name : str
            The name of the tokenizer to use.
        external_name : Optional[str], default=None
            The external name of the dataset. If None, it is the same as the root
            directory.
        format_ : str, default="SMILES"
            The format of the text file. Currently only "SMILES" is supported.
        size : Optional[int], default=None
            The number of data points to load. If None, all data points are loaded.
        masked_fractions : Optional[dict[str, float]], default=None
            A dictionary of masked fractions for each atom type. If None, all atoms
            are unmasked.
        max_smiles_len : int, default=300
            The maximum length of the SMILES string.
        vocab_size : Optional[int], default=None
            The size of the vocabulary. If None, it is the same as the size of the tokenizer.

        Attributes
        ----------
        root : Path
            The root directory of the dataset.
        tokenizer : Tokenizer
            The tokenizer used to tokenize the strings.
        PAD_TOKEN : str
            The token used to pad the strings.

        """
        self.vocab_size = vocab_size
        format_ = format_.lower()
        self.format = format_
        root = Path(root)
        self.root = root / format_
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.external_name = external_name
        self.max_smiles_len = max_smiles_len
        try:
            self.tokenizer: Tokenizer = load_tokenizer(tokenizer_name + "_" + self.external_name)
            self.tokenizer_name = tokenizer_name + "_" + self.external_name
        except FileNotFoundError:
            self.tokenizer: Tokenizer = load_tokenizer(tokenizer_name)
            self.tokenizer_name = tokenizer_name
        if masked_fractions is None:
            masked_fractions = {"nodes": 0.15, "edges": 0.05}
        self.masked_fractions = masked_fractions
        self.size = size
        torch.utils.data.Dataset.__init__(self)
        succesfully_loaded = self.load()
        if not succesfully_loaded:
            graphchem_root = str(root) if not str(root).endswith("_txt") else str(root)[:-4]
            graphchem = GraphChem(
                root=graphchem_root,
                size=size,
                masked_fractions=masked_fractions,
                pre_filter=AllPreFilters(
                    [
                        PreFilterRemoveLongSMILES(self.max_smiles_len),
                        PreFilterCheckValidSMILES(),
                    ],
                ),
            )
            graphchem.split()
            graphchem.maybe_save_as_txt(format_)
            self.load()

    def load(self) -> bool:
        """Loads the dataset from the raw data.

        If the raw data does not exist, the function returns False. If the
        tokenizer is not trained, the function trains the tokenizer. Then, the
        function loads the raw data and masks it according to the specified
        masked fractions. The masked and true data are stored in the `data`
        attribute of the class.

        Returns
        -------
        bool
            True if the dataset is loaded successfully, False otherwise.

        """
        if not (self.raw_dir / "data.txt").exists():
            return False
        if not self.tokenizer.trained:
            self.train_tokenizer()
        self.data = []
        with (self.raw_dir / "data.txt").open("r", encoding="utf-8") as f:
            lines = f.readlines()
            for line_full in lines:
                line = line_full.strip("\n")
                if line and len(line) <= self.max_smiles_len:
                    encoded = self.tokenizer.encode(line.strip())
                    ids = torch.tensor(encoded.ids, dtype=torch.long)
                    n_ids_to_mask = int(ids.shape[0] * self.masked_fractions["nodes"])
                    if n_ids_to_mask == 0:
                        n_ids_to_mask = 1
                    to_mask = torch.randperm(ids.shape[0])[:n_ids_to_mask]
                    masked_ids = ids.clone()
                    masked_ids[to_mask] = self.tokenizer.token_to_id(self.MASK_TOKEN)
                    self.data.append({"ids": ids, "masked_ids": masked_ids})

        return True

    def train_tokenizer(self) -> None:
        """Trains the tokenizer on the raw data.

        The tokenizer is trained using the parameters specified in the
        `get_trainer_params` function. The trained tokenizer is then saved to a
        file in the `models/weights/tokenizers` directory with the same name as
        the tokenizer but with the external name appended to it. The path to the
        file is also saved in the `models/config/tokenizers` directory.

        """
        trainer = BpeTrainer(**OmegaConf.to_container(get_trainer_params(self.tokenizer_name)))
        self.tokenizer.train([str((self.raw_dir / "data.txt").absolute().resolve())], trainer)
        self.tokenizer.add_special_tokens([self.PAD_TOKEN, self.MASK_TOKEN])
        self.tokenizer.trained = True
        tokenizer_config = load_tokenizer_config(self.tokenizer_name)
        self.tokenizer_name = self.tokenizer_name + "_" + self.external_name
        tokenizer_config.path = str(
            (Path("models") / "weights" / "tokenizers" / f"{self.tokenizer_name}.json")
            .absolute()
            .relative_to(Path.cwd()),
        )
        (Path("models") / "weights" / "tokenizers").mkdir(exist_ok=True, parents=True)
        self.tokenizer.save(tokenizer_config.path)
        OmegaConf.save(
            tokenizer_config,
            Path("models") / "configs" / "tokenizers" / f"{self.tokenizer_name}.yaml",
        )

    def __getitem__(self, idx: int | slice) -> dict[str, str]:
        """Gets the idx-th item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item to be retrieved.

        Returns
        -------
        torch.Tensor
            The idx-th item from the dataset.

        """
        return self.data[idx]

    def collate_fn(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Collate function for the GraphChem dataset.

        Parameters
        ----------
        batch : list of str
            A list of SMILES strings.

        Returns
        -------
        list of list of int
            A list of lists of token indices, where each inner list corresponds to a SMILES string.

        """
        masked = []
        targets = []
        for item in batch:
            masked.append(item["masked_ids"])
            targets.append(item["ids"])
        pad_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        res = {
            "masked_ids": torch.nn.utils.rnn.pad_sequence(
                sequences=masked,
                batch_first=True,
                padding_value=pad_id,
            ),
            "ids": torch.nn.utils.rnn.pad_sequence(
                sequences=targets,
                batch_first=True,
                padding_value=pad_id,
            ),
        }
        res["attention_mask"] = res["masked_ids"] != pad_id
        return res

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.data)
