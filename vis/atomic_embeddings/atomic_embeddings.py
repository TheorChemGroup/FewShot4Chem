"""Main function to plot atomic embeddings is defined here."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import deeplake
import umap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tqdm.auto import tqdm

from utils.minor import safe_write_image
from vis.atomic_embeddings.boundaries import reduce_and_plot_boundaries
from vis.atomic_embeddings.dimension_reduction import plot_pca, plot_umap
from vis.atomic_embeddings.metrics import plot_and_save_metrics
from vis.atomic_embeddings.utils import (
    idx_2_atom,
    recursively_merge_dicts,
    recursively_set_values,
)
from vis.constants import MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS
from vis.pieces import plot_confusion_matrix

logger = logging.getLogger(__name__)


def plot_atomic_embeddings(  # noqa: PLR0913
    model_name: str,
    dataset_name: str,
    svcs: Sequence[SVC],
    svc_names: Sequence[str],
    figure_width: dict | int | None = None,
    figure_height: dict | int | None = None,
) -> None:
    """Plots atomic embeddings for a given model and dataset, and saves the results.

    Parameters
    ----------
    model_name : str
        The name of the model.
    dataset_name : str
        The name of the dataset.
    svcs : Sequence[SVC]
        The support vector machines to be plotted.
    svc_names : Sequence[str]
        The names of the support vector machines.
    figure_width : dict | Optional[int], optional
        The width of the figures to be plotted. If an int, sets all widths to this value.
        If a dict, sets the widths to the values in the dict. If None, uses default values.
    figure_height : dict | Optional[int], optional
        The height of the figures to be plotted. If an int, sets all heights to this value.
        If a dict, sets the heights to the values in the dict. If None, uses default values.

    """
    figure_width_default = {
        "umap": 1400,
        "pca": 1400,
        "metrics": 1400,
        "boundaries": {
            "umap": 600,
            "pca": 600,
        },
        "confusion_matrix": 600,
    }
    figure_height_default = {
        "umap": 1000,
        "pca": 1000,
        "metrics": 1000,
        "boundaries": {
            "umap": 600,
            "pca": 600,
        },
        "confusion_matrix": 600,
    }
    if figure_width is None:
        figure_width = figure_width_default
    elif isinstance(figure_width, int):
        recursively_set_values(figure_width_default, figure_width)
        figure_width = figure_width_default
    else:
        figure_width = recursively_merge_dicts(figure_width_default, figure_width)

    if figure_height is None:
        figure_height = figure_height_default
    elif isinstance(figure_height, int):
        recursively_set_values(figure_height_default, figure_height)
        figure_height = figure_height_default
    else:
        figure_height = recursively_merge_dicts(figure_height_default, figure_height)

    atomic_embeddings_storage = deeplake.open_read_only(
        f"results/{dataset_name}/{model_name}/atomic_embeddings",
    )
    embeddings = atomic_embeddings_storage["embeddings"][:MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS]
    metrics_file_path = Path("results") / dataset_name / model_name / "metrics.json"
    with metrics_file_path.open(encoding="utf-8") as f:
        metrics = json.load(f)
    plot_umap(
        atomic_embeddings_storage,
        embeddings,
        dataset_name,
        model_name,
        figure_width["umap"],
        figure_height["umap"],
    )
    plot_pca(
        atomic_embeddings_storage,
        embeddings,
        dataset_name,
        model_name,
        figure_width["pca"],
        figure_height["pca"],
    )
    plot_and_save_metrics(
        metrics,
        dataset_name,
        model_name,
        figure_width["metrics"],
        figure_height["metrics"],
    )

    if logger.isEnabledFor(logging.INFO):
        progress_bar = tqdm(
            zip(svcs, svc_names, strict=False),
            desc="Plotting boundaries and confusion matrices",
            total=len(svcs),
        )
    else:
        progress_bar = zip(svcs, svc_names, strict=False)

    for svc, svc_name in progress_bar:
        reduce_and_plot_boundaries(
            svc,
            embeddings,
            atomic_embeddings_storage["atomic_id"][:MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS],
            umap.UMAP(n_neighbors=30),
            f"results/{dataset_name}/{model_name}/plots",
            svc_name,
            "umap",
            figure_width["boundaries"]["umap"],
            figure_height["boundaries"]["umap"],
        )
        reduce_and_plot_boundaries(
            svc,
            embeddings,
            atomic_embeddings_storage["atomic_id"][:MAX_ITEMS_FOR_ATOMIC_EMBEDDINGS],
            PCA(),
            f"results/{dataset_name}/{model_name}/plots",
            svc_name,
            "pca",
            figure_width["boundaries"]["pca"],
            figure_height["boundaries"]["pca"],
        )
        cm_file_path = (
            Path("results") / dataset_name / model_name / f"{svc_name}_confusion_matrix.json"
        )
        with cm_file_path.open(
            encoding="utf-8",
        ) as f:
            cm = json.load(f)
        confusion_matrix_fig = plot_confusion_matrix(
            cm["matrix"],
            classes=[idx_2_atom[atomic_id] for atomic_id in cm["classes"]],
        )
        safe_write_image(
            confusion_matrix_fig,
            f"results/{dataset_name}/{model_name}/plots/{svc_name}_confusion_matrix.png",
        )
