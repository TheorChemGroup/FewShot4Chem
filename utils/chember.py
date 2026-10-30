"""provides a wrapper for PubChem77M dataset with Chemberta2 embeddings."""

import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch_geometric
from omegaconf import OmegaConf
from rdkit import Chem
from torch_geometric.data import Data, OnDiskDataset
from tqdm.auto import tqdm

from utils.constants import ATOM_2_IDX
from utils.embed_dataset import EmbeddingParams, embed_dataset
from utils.load_models import load_model

logger = logging.getLogger(__name__)


def find_threshold(dataset: OnDiskDataset, samples: float, frac: float = 1) -> float:
    """Find a threshold for a given dataset to use with Chember.

    Finds a threshold for a given dataset by sampling a subset of the dataset,
    computing the mean distance between each pair of samples, and then multiplying
    this mean distance by a fraction (default is 1.0).

    Parameters
    ----------
    dataset : OnDiskDataset
        The dataset to find the threshold for.
    samples : float or int
        The number of samples to use for finding the threshold. If a float, will be
        interpreted as a fraction of the dataset size.
    frac : float, optional
        The fraction of the mean distance to use as the threshold, by default 1.0.

    Returns
    -------
    float
        The threshold to use.

    """
    logger.info("Finding threshold...")

    if isinstance(samples, float):
        samples = int(samples * len(dataset))

    indices = torch.randperm(len(dataset))[:samples].tolist()

    x = torch.stack([dataset[idx].y for idx in indices], dim=0)

    distances = torch.cdist(x, x, p=2)

    triu_indices = torch.triu_indices(samples, samples, offset=1)
    pair_distances = distances[triu_indices[0], triu_indices[1]]

    mean_distance = pair_distances.mean().item()
    return frac * mean_distance


class Chember(OnDiskDataset):
    """Chember dataset.

    Chember is a dataset of all molecules in the PubChem database, embedded using
    ChemBERTa-2. This dataset is huge, so it is stored on disk.
    """

    MAX_MOLECULES = 9900000
    BUFFER_SIZE = 16384
    EMBED_CALC_BUFFER_SIZE = 8192

    def process(self) -> None:  # noqa: C901, PLR0912, PLR0915 # TODO: break down into smaller functions
        """Process the raw data and generate the Chember dataset.

        This method is a bit of a hack, as it doesn't actually use the `OnDiskDataset`
        methods, but it does use the `OnDiskDataset` abstract methods.

        It embeds the raw data using a graph neural network and saves the embeddings
        to disk in a directory specified by `processed_dir`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        file_logger = logging.getLogger("chember")
        file_logger.setLevel(logger.getEffectiveLevel())
        file_handler = logging.FileHandler(Path(self.processed_dir) / "process.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s|%(name)s| %(levelname)s: %(message)s"),
        )
        file_logger.addHandler(file_handler)
        with Path(self.raw_paths[0]).open(encoding="utf-8") as raw_data_file:
            smiles_list = raw_data_file.readlines()

        try:
            with Path(self.processed_paths[1]).open(encoding="utf-8") as f:
                embedded_idx = set(json.load(f))
            if len(embedded_idx) != len(self):
                logger.warning(
                    "Length of embedded_idx (%d) does not match length of the dataset (%d)!"
                    "This is probably a bug caused by early-stopped processing.",
                    len(embedded_idx),
                    len(self),
                )
                raise FileNotFoundError  # noqa: TRY301
        except (FileNotFoundError, json.JSONDecodeError):
            logger.debug("Getting idx for already embedded molecules...")
            progress_bar = tqdm(self) if logger.isEnabledFor(logging.DEBUG) else self
            embedded_idx = {data.idx for data in progress_bar}

        logger.info("Already embedded %d molecules", len(embedded_idx))

        chemberta_config = OmegaConf.load("models/configs/chemberta/chemberta.yaml")

        chemberta = load_model(chemberta_config)

        subset = []
        (Path(self.processed_dir) / "embeddings").mkdir(parents=True, exist_ok=True)
        if logger.isEnabledFor(logging.INFO):
            progress_bar = tqdm(smiles_list[len(embedded_idx) :])
            progress_bar.update(len(embedded_idx))
        else:
            progress_bar = smiles_list[len(embedded_idx) :]

        buffer = []
        try:
            for idx, smiles in enumerate(progress_bar):
                if idx not in embedded_idx:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            mol = Chem.AddHs(mol)

                            adj_matrix = Chem.GetAdjacencyMatrix(mol)

                            atom_idxs = [ATOM_2_IDX[atom.GetSymbol()] for atom in mol.GetAtoms()]
                            atom_idxs = torch.tensor(atom_idxs, dtype=torch.long)

                            edge_idxs = (
                                torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
                                .t()
                                .contiguous()
                            )
                            subset.append(
                                Data(
                                    x=atom_idxs,
                                    edge_index=edge_idxs.T,
                                    smiles=smiles,
                                    idx=idx,
                                ),
                            )

                    except KeyError:
                        continue

                if len(subset) >= self.EMBED_CALC_BUFFER_SIZE or idx == len(smiles_list) - 1:
                    try:
                        emb_path = Path(self.processed_dir) / "embeddings" / f"embeddings_{idx}.npy"
                        embeddings = embed_dataset(
                            chemberta,
                            subset,
                            embedding_params=EmbeddingParams(
                                batch_size=chemberta_config.batch_size,
                                path=emb_path,
                            ),
                            store_intermediates=True,
                        )

                    except Exception as e:
                        if re.search("The expanded size of the tensor", str(e)):
                            file_logger.exception()
                        else:
                            raise
                    else:
                        for datum, embedding in zip(subset, embeddings, strict=False):
                            datum.y = torch.tensor(embedding)
                            embedded_idx.add(datum.idx)
                            buffer.append(datum)
                        self.extend(buffer)
                        buffer = []
                        emb_path.unlink()
                        if len(self) > self.MAX_MOLECULES:
                            break
                    subset = []
        finally:
            with Path(self.processed_paths[1]).open("w", encoding="utf-8") as f:
                json.dump(list(embedded_idx), f)

        with Path(self.processed_paths[-1]).open("w", encoding="utf-8") as f:
            f.write("Curiosity, noun\nA strong desire to know or learn something.")

    @property
    def processed_file_names(self) -> str:
        """Returns the processed file names."""
        return ["sqlite.db", "embedded_idx.json", "I_AM_PROCESSED"]

    @property
    def raw_file_names(self) -> str:
        """Returns the raw file names."""
        return "pubchem77M.txt"


class ChemberWithNegative(OnDiskDataset):
    """A Chember dataset with negative samples.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    chember_or_path : Chember | str
        Path to the Chember dataset or a Chember dataset object.
    distance : str | callable, optional
        Distance metric to use. Can be "cosine", "euclidean", "l2", "mse", or a callable.
        (default: "cosine")
    threshold : Optional[float], optional
        Threshold value for considering two molecules as negative samples.
        If None, the threshold is determined based on the samples_to_find_dist parameter.
        (default: None)
    samples_to_find_dist : float | str, optional
        The number of samples to use for determining the threshold value.
        If str, should be "all".
        (default: 0.2)
    max_iter : float, optional
        The maximum number of iterations to find negative samples.
        (default: float("inf"))
    transform : Optional[Callable], optional
        A function/transform that takes in a Data object and returns a transformed version.
        (default: None)
    pre_filter : Optional[Callable], optional
        A function that takes in an object and returns True if the object should be kept.
        (default: None)
    buffer_size : Optional[int], optional
        The size of the buffer to use for storing molecules before writing them to disk.
        If None, the buffer size is determined based on the RAM available.
        (default: None)

    Notes
    -----
    The threshold is determined by sampling molecules from the dataset and computing the distance
    between them.
    The number of samples to use for determining the threshold value is determined by the
    samples_to_find_dist parameter.
    If samples_to_find_dist is "all", all molecules in the dataset are used.
    The maximum number of iterations to find negative samples is determined by the max_iter
    parameter.
    If max_iter is reached, the resulting negative samples may still be close to positive.

    """

    def __init__(  # noqa: PLR0913 #TODO: add config to reduce argument count
        self,
        root: str,
        chember_or_path: Chember | str,
        distance: str | Callable = "cosine",
        threshold: float | None = None,
        samples_to_find_dist: float | str = 0.2,
        max_iter: float = float("inf"),
        transform: Callable | None = None,
        pre_filter: Callable | None = None,
        buffer_size: int | None = None,
    ) -> None:
        """Initializes a ChemberWithNegative dataset.

        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        chember_or_path : Chember | str
            Path to the Chember dataset or a Chember dataset object.
        distance : str | callable, optional
            Distance metric to use. Can be "cosine", "euclidean", "l2", "mse", or a callable.
            (default: "cosine")
        threshold : Optional[float], optional
            Threshold value for considering two molecules as negative samples.
            If None, the threshold is determined based on the samples_to_find_dist parameter.
            (default: None)
        samples_to_find_dist : float | str, optional
            The number of samples to use for determining the threshold value.
            If str, should be "all".
            (default: 0.2)
        max_iter : float, optional
            The maximum number of iterations to use for determining the threshold value.
            (default: inf)
        transform : Optional[Callable], optional
            A function/transform that takes in an object and returns a transformed version.
            The data object will be transformed before every access.
            (default: None)
        pre_filter : Optional[Callable], optional
            A function/transform that takes in an object and returns a transformed version.
            The data object will be transformed before being saved to disk.
            (default: None)
        buffer_size : Optional[int], optional
            The buffer size to use when processing the Chember dataset.
            If None, the buffer size of the Chember dataset is used.
            (default: None)

        """
        self.root = root
        if isinstance(distance, str):
            distance = distance.lower()
        if distance == "cosine":
            self.distance = torch.nn.CosineSimilarity(dim=1)
        elif distance in {"euclidean", "l2", "mse"}:
            self.distance = torch.linalg.norm
        elif isinstance(distance, callable):
            self.distance = distance
        else:
            raise ValueError(distance)
        try:
            with Path(self.processed_paths[1]).open("r", encoding="utf-8") as f:
                current_params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_params = {}
        if isinstance(chember_or_path, Chember):
            self.chember = chember_or_path
        else:
            self.chember = Chember(chember_or_path)
        if buffer_size is None:
            self.BUFFER_SIZE = self.chember.BUFFER_SIZE
        self.threshold = threshold
        if self.threshold is None:
            self.threshold = current_params.get("threshold")
        self.samples_to_find_dist = samples_to_find_dist
        self.max_iter = max_iter
        self.permutation = {}
        super().__init__(root, transform, pre_filter)

    def process(self) -> None:
        """Process the Chember dataset.

        This involves finding the threshold for the similarity measure,
        and then finding the negative samples for each molecule in the dataset.

        The negative samples are found by sampling a random molecule from the dataset,
        and checking if the distance between the two molecules is greater than the threshold.
        If the distance is greater than the threshold, or if the maximum number of iterations
        has been reached,
        the negative sample is added to the dataset and the iteration is stopped.

        The process is repeated until all molecules in the dataset have been processed.

        Returns:
            None

        """
        self.threshold = (
            self.threshold
            if self.threshold is not None
            else find_threshold(self.chember, self.samples_to_find_dist)
        )
        try:
            with Path(self.processed_paths[1]).open("r", encoding="utf-8") as f:
                current_params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_params = {}
        current_params["threshold"] = self.threshold
        with Path(self.processed_paths[2]).open("w", encoding="utf-8") as f:
            json.dump(current_params, f)

        chember_len = len(self.chember)

        got_no_negative = set(range(chember_len))
        iters = 0
        if logger.isEnabledFor(logging.INFO):
            progress_bar = tqdm(total=chember_len, desc="Finding negative samples")
        else:
            progress_bar = None
        buffer = []
        while got_no_negative:
            if iters >= self.max_iter:
                logger.warning(
                    "Did not find negative samples for %d molecules in %d iterations,"
                    "the resulting negative samples may still be close to positive",
                    len(got_no_negative),
                    iters,
                )
                break
            found_negative = set()
            for datum_idx, negative_idx in zip(
                got_no_negative,
                torch.randint(0, chember_len, size=(len(got_no_negative),)),
                strict=False,
            ):
                datum = self.chember[datum_idx]
                negative_sample = self.chember[negative_idx]
                dist = self.distance(negative_sample.y - datum.y)
                if dist > self.threshold or iters + 1 >= self.max_iter:
                    found_negative.add(datum_idx)
                    self.permutation[datum_idx] = len(self)
                    buffer.append(negative_sample)
                    if progress_bar is not None:
                        progress_bar.update(1)
                    if len(buffer) >= self.BUFFER_SIZE or len(got_no_negative) == len(
                        found_negative,
                    ):
                        self.extend(buffer)
                        buffer = []
            got_no_negative = got_no_negative - found_negative
            iters += 1

    def get(self, idx: int) -> torch_geometric.data.Data:
        """Gets a molecule from the dataset by its index.

        Parameters
        ----------
        idx: int
            The index of the molecule to retrieve.

        Returns
        -------
        chember_value: torch_geometric.data.Data
            The molecule at the given index.

        """
        idx = self.permutation.get(idx, idx)
        negative = super().get(idx).negative
        chember_value = self.chember[idx]
        chember_value.negative = negative
        return chember_value

    def multi_get(self, idxs: Sequence[int]) -> list[torch_geometric.data.Data]:
        """Gets multiple molecules from the dataset by their indices.

        Parameters
        ----------
        idxs : list[int]
            The indices of the molecules to retrieve.

        Returns
        -------
        list[torch_geometric.data.Data]
            The molecules at the given indices.

        """
        idxs = [self.permutation.get(idx, idx) for idx in idxs]
        negatives = [datum.negative for datum in super().multi_get(idxs)]
        res = []
        for negative, datum in zip(negatives, self.chember.multi_get(idxs), strict=False):
            datum.negative = negative
            res.append(datum)
        return res

    @property
    def processed_file_names(self) -> str:
        """Returns the processed file names."""
        return ["sqlite.db", "params.json"]
