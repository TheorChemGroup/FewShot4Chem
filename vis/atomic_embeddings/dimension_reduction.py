"""Function to plot dimension reduction of atomic embeddings."""

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import plotly.express as px
import umap
from sklearn.decomposition import PCA

from utils.minor import safe_write_image
from vis.atomic_embeddings.utils import (
    idx_2_atom,
)
from vis.constants import MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS
from vis.defaults import DEFAULT_STYLE

logger = logging.getLogger(__name__)


def plot_umap(
    atomic_embeddings_storage: dict,
    embeddings: np.ndarray,
    dataset_name: str,
    model_name: str,
    *,
    style: dict = DEFAULT_STYLE,
) -> None:
    """Plot the UMAP-reduced atomic embeddings.

    Parameters
    ----------
    atomic_embeddings_storage : dict
        The atomic embeddings storage.
    embeddings : np.ndarray
        The atomic embeddings.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    style : dict, optional
        The style of the plot, by default DEFAULT_STYLE (from vis.defaults)

    """
    umap_reducer = umap.UMAP(n_components=2, n_neighbors=30)

    embeddings_2d = umap_reducer.fit_transform(embeddings)

    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=[
            idx_2_atom[atomic_id]
            for atomic_id in atomic_embeddings_storage["atomic_id"][
                :MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS
            ]
        ],
    )
    style = deepcopy(style)
    style.update(
        title="UMAP-reduced atomic embeddings",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
    )

    fig.update_layout(**style)

    save_path = Path("results") / dataset_name / model_name / "plots" / "atomic_embeddings_umap.png"

    safe_write_image(fig, save_path)

    logger.info(
        "UMAP-reduced atomic embeddings plot saved to %s",
        save_path,
    )


def plot_pca(
    atomic_embeddings_storage: dict,
    embeddings: np.ndarray,
    dataset_name: str,
    model_name: str,
    *,
    style: dict = DEFAULT_STYLE,
) -> None:
    """Plot the PCA-reduced atomic embeddings.

    Parameters
    ----------
    atomic_embeddings_storage : dict
        The atomic embeddings storage.
    embeddings : np.ndarray
        The atomic embeddings.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    style : dict, optional
        The style of the plot, by default DEFAULT_STYLE (from vis.defaults)

    """
    pca_reducer = PCA(n_components=2)

    embeddings_2d = pca_reducer.fit_transform(embeddings)

    fig = px.scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color=[
            idx_2_atom[atomic_id]
            for atomic_id in atomic_embeddings_storage["atomic_id"][
                :MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS
            ]
        ],
    )

    style = deepcopy(style)
    style.update(
        {
            "title": "PCA-reduced atomic embeddings",
            "xaxis_title": "PCA 1",
            "yaxis_title": "PCA 2",
        },
    )

    fig.update_layout(**style)

    save_path = Path("results") / dataset_name / model_name / "plots" / "atomic_embeddings_pca.png"
    safe_write_image(fig, save_path)

    logger.info("PCA-reduced atomic embeddings plot saved to %s", save_path)
