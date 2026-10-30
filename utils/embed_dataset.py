"""Provides a tool for embedding a dataset using a pytorch model."""

import contextlib
import fcntl
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch_geometric
from torch import nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def find_proper_path_for_embeddings(
    path: str | Path | None,
    dataset: Sequence[torch_geometric.data.Data],
) -> Path:
    """Find a proper path to save the embeddings to.

    If the path is provided, it is converted to a Path object if it is not already one.
    If the path is None and the dataset is a torch_geometric.data.Dataset, the path
    is set to the root of the dataset plus "embeddings.npy". If the path is still None,
    it is returned as is.

    Parameters
    ----------
    path : Optional[Union[str, Path]]
        The path to save the embeddings to. If None, will try to infer it from the dataset.
    dataset : Sequence[torch_geometric.data.Data]
        The dataset to embed.

    Returns
    -------
    Path
        The path to save the embeddings to.

    """
    if isinstance(path, str):
        path = Path(path)
    if path is None and isinstance(dataset, torch_geometric.data.Dataset):
        root = dataset.root
        path = (
            None if root == torch_geometric.data.dataset.MISSING else Path(root) / "embeddings.npy"
        )
    return path


class EmbeddingParams:
    """Class for embedding parameters.

    This is an utility for embed_dataset function.
    """

    path: str | Path | None
    batch_size: int
    store_intermediates: bool
    recalculate: bool
    return_embeddings: bool

    def __init__(
        self,
        path: str | Path | None = None,
        batch_size: int = 128,
        *,
        store_intermediates: bool = False,
        recalculate: bool = False,
        return_embeddings: bool = True,
    ) -> None:
        """Initialize an EmbeddingParams object.

        Parameters
        ----------
        path : Optional[Union[str, Path]], optional
            Path to store the embeddings, by default None
        batch_size : int, optional
            Batch size for the embedding, by default 128
        store_intermediates : bool, optional
            Whether to store intermediate embeddings, by default False
        recalculate : bool, optional
            Whether to recalculate the embeddings if they already exist, by default False
        return_embeddings : bool, optional
            Whether to return the embeddings, by default True

        """
        self.path = path
        self.batch_size = batch_size
        self.store_intermediates = store_intermediates
        self.recalculate = recalculate
        self.return_embeddings = return_embeddings


def embed_dataset(
    model: nn.Module,
    dataset: Sequence[torch_geometric.data.Data],
    embedding_params: EmbeddingParams,
) -> None | np.ndarray:
    """Embed a dataset using a pytorch model.

    Args:
        model: model to use for embedding
        dataset: dataset to embed
        embedding_params: parameters for embedding. Created with EmbeddingParams class. May contain:
            + `path`: path to save embeddings to. If `None`, will try to infer it from dataset.
            If failed, will not save embeddings.
            + `batch_size`: batch size to use for embedding
            + `store_intermediates`: whether to store intermediates
            + `recalculate`: whether to recalculate embeddings
            + `return_embeddings`: whether to return embeddings

    Returns:
        embeddings: embeddings of the dataset, either loaded from the path if it exists or computed

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = find_proper_path_for_embeddings(embedding_params.path, dataset)
    lock_fd = None
    if path is None:
        logger.warning("Could not find out the path to save the embeddings")
    elif path.suffix == ".npy":
        if not embedding_params.recalculate:
            with contextlib.suppress(FileNotFoundError):
                res = np.load(path)
                logger.info("Embeddings found at %s, skipping calculation", path)
                return res
        # Acquire exclusive lockfile to serialize concurrent (model, dataset)
        # cache-misses across SLURM tasks. After the lock, re-check the cache —
        # another task may have populated it while we waited.
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".npy.lock")
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not embedding_params.recalculate:
            with contextlib.suppress(FileNotFoundError):
                res = np.load(path)
                logger.info("Embeddings populated by peer at %s, skipping calculation", path)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                return res
    else:
        logger.warning(
            "Resulting path has extension %s, expected '.npy'",
            path.suffix,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=embedding_params.batch_size,
        collate_fn=torch_geometric.data.Batch.from_data_list,
        shuffle=False,
    )

    if embedding_params.store_intermediates:
        tempdir = TemporaryDirectory(prefix="temp/")
        storage = torch_geometric.data.OnDiskDataset(root=tempdir.name)
    else:
        storage = []

    if logger.isEnabledFor(logging.INFO):
        progress_bar = tqdm(dataloader, desc="Calculating embeddings")
    else:
        progress_bar = dataloader

    previous_mode = getattr(model, "mode", None)
    if previous_mode is not None:
        model.mode = "embedding"
    model.eval()

    model.to(device)
    with torch.inference_mode():
        for data in progress_bar:
            storage.append(torch_geometric.data.Data(y=model(data.to(device)).cpu()))
    if previous_mode is not None:
        model.mode = previous_mode

    if embedding_params.return_embeddings:
        logger.info("Concatenating batched embeddings...")
        res = np.concatenate([data.y.numpy() for data in storage])
        if path is not None and path.suffix == ".npy":
            # Atomic write: np.save to a tempfile in the same dir, then os.replace.
            # Prevents partial-file reads if a peer races on the same path.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy.tmp")
            os.close(fd)
            try:
                np.save(tmp, res)
                # np.save appends ".npy"; the actual file is tmp + ".npy".
                os.replace(tmp + ".npy", str(path))
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                with contextlib.suppress(OSError):
                    os.unlink(tmp + ".npy")
        else:
            np.save(path, res)
    else:
        res = None
    if lock_fd is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    if embedding_params.store_intermediates:
        del storage
        tempdir.cleanup()
    return res
