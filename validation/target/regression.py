"""Regression validator using the Head abstraction."""

import logging
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch_geometric.data import Data
from tqdm.auto import tqdm

from utils.datasets import TARGETS_DESCRIPTIONS
from utils.minor import safe_write_image
from validation.heads import Head, load_head_from_config
from validation.target.abstract import TargetValidator
from vis import plot_pred_vs_true_and_metric
from vis.constants import MAX_ITEMS_FOR_METRICS
from vis.defaults import DEFAULT_STYLE

logger = logging.getLogger(__name__)


class RegressionValidator(TargetValidator):
    metric_functions: ClassVar[dict[str, Callable]] = {
        "R²": partial(r2_score, multioutput="raw_values"),
        "MAE": partial(mean_absolute_error, multioutput="raw_values"),
        "MSE": partial(mean_squared_error, multioutput="raw_values"),
        "MAPE": partial(mean_absolute_percentage_error, multioutput="raw_values"),
    }
    metrics_to_display: ClassVar[set[str]] = {"R²", "MAE", "MSE", "MAPE"}

    def __init__(
        self,
        head_config: OmegaConf,
        dataset: Sequence[Data],
        *,
        random_seed: int = 42,
        dataset_name: str | None = None,
    ) -> None:
        super().__init__(head_config, dataset, random_seed, dataset_name)

    def create_head(self) -> Head:
        return load_head_from_config(self.head_config)

    def plot_results_for_run(
        self,
        model_config: OmegaConf,
        targets: np.ndarray,
        predictions: np.ndarray,
        metrics: dict[str, Any],
        run_id: int,
        *,
        style: dict[str, Any] = DEFAULT_STYLE,
        single_atom: bool = False,
    ) -> None:
        if single_atom:
            raise NotImplementedError
        plots_path = (
            self._result_root(model_config) / "plots" / str(run_id)
        )
        descriptions = TARGETS_DESCRIPTIONS[self.dataset_name]
        true = targets[:MAX_ITEMS_FOR_METRICS]
        pred = predictions[:MAX_ITEMS_FOR_METRICS]
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if true.ndim == 1:
            true = true.reshape(-1, 1)
        progress = tqdm(descriptions, desc="Generating plots") if logger.isEnabledFor(logging.INFO) else descriptions
        for i, description in enumerate(progress):
            for metric_name, values in metrics.items():
                folder = plots_path / metric_name
                fig = make_subplots(rows=1, cols=1)
                plot_pred_vs_true_and_metric(
                    pred[:, i],
                    true[:, i],
                    fig,
                    metric_value=values[i],
                    metric_name=metric_name,
                    title=description,
                    style=style,
                )
                safe_write_image(fig, folder / f"{i:03d}. {description}.png")
