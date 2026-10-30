"""Classification validator using the Head abstraction."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Data
from umap import UMAP

from utils.datasets import CLASSES_NAMES, TARGETS_NAMES
from utils.minor import safe_write_image
from validation.heads import Head, load_head_from_config
from validation.target.abstract import TargetValidator
from validation.utils import calculate_sample_weights, to_multitarget_metric
from vis import (
    plot_classification_results_with_dimension_reduction,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from vis.defaults import DEFAULT_STYLE

logger = logging.getLogger(__name__)


class ClassificationValidator(TargetValidator):
    metric_functions: ClassVar[dict[str, Callable]] = {
        "ROC_AUC": to_multitarget_metric(roc_auc_score, uses_probabilities=True),
        "PR_AUC": to_multitarget_metric(average_precision_score, uses_probabilities=True),
        "accuracy": to_multitarget_metric(accuracy_score, uses_probabilities=False),
        "F1": to_multitarget_metric(f1_score, uses_probabilities=False),
        "precision": to_multitarget_metric(precision_score, uses_probabilities=False),
        "recall": to_multitarget_metric(recall_score, uses_probabilities=False),
        "confusion_matrix": to_multitarget_metric(confusion_matrix, uses_probabilities=False),
        "roc_curve": to_multitarget_metric(roc_curve, uses_probabilities=True, return_list=True),
        "precision_recall_curve": to_multitarget_metric(
            precision_recall_curve,
            uses_probabilities=True,
            return_list=True,
        ),
    }
    metrics_to_display: ClassVar[set[str]] = {
        "ROC_AUC",
        "PR_AUC",
        "accuracy",
        "F1",
        "precision",
        "recall",
    }

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

    def get_fit_kwargs(
        self,
        x_train: np.ndarray,  # noqa: ARG002
        x_val: np.ndarray,  # noqa: ARG002
        y_train: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        # Weights computed on already-carved partitions (validator does the carve before calling
        # this method), so lengths match what each head receives in head.fit(x_train_h, y_train_h).
        sample_weight = calculate_sample_weights(y_train)
        sample_weight_eval_set = calculate_sample_weights(y_val)
        return {
            "sample_weight": sample_weight,
            "sample_weight_eval_set": (sample_weight_eval_set,),
        }

    def plot_results_for_run(
        self,
        model_config: OmegaConf,
        embeddings: np.ndarray,
        targets: np.ndarray,
        predictions: np.ndarray,
        metrics: dict[str, Any],
        run_id: int,
        *,
        style: dict[str, Any] = DEFAULT_STYLE,
    ) -> None:
        target_names = TARGETS_NAMES[self.dataset_name]
        save_path = self._result_root(model_config) / "plots" / str(run_id)
        save_path.mkdir(parents=True, exist_ok=True)
        models = (
            ("UMAP", UMAP(n_components=2, random_state=self.random_seed)),
            ("PCA", PCA(n_components=2)),
        )
        for name, model in models:
            for i, target in enumerate(target_names):
                fig = plot_classification_results_with_dimension_reduction(
                    features=embeddings,
                    targets=targets[:, i],
                    predictions=predictions[:, i],
                    dimension_reduction_model=model,
                    dimension_reduction_model_name=name,
                    style=style,
                )
                safe_write_image(fig, save_path / name / f"{i:02}. {target}.png")
        for i, target in enumerate(target_names):
            cm = metrics["confusion_matrix"][i]
            fig = plot_confusion_matrix(
                cm,
                classes=CLASSES_NAMES[self.dataset_name][i],
                style=style,
                metrics={
                    n: self.formatters.get(n, lambda x: f"{x:.3f}")(metrics[n][i])
                    for n in ["accuracy", "precision", "recall", "F1"]
                },
            )
            fig.update_layout(title=target)
            safe_write_image(fig, save_path / "confusion_matrix" / f"{i:02}. {target} Confusion Matrix.png")
            fpr, tpr, _ = metrics["roc_curve"][i]
            fig = plot_roc_curve(fpr, tpr, auc=metrics["ROC_AUC"][i], style=style)
            fig.update_layout(title=target)
            safe_write_image(fig, save_path / "roc_curve" / f"{i:02}. {target} ROC Curve.png")
            precision, recall, _ = metrics["precision_recall_curve"][i]
            fig = plot_precision_recall_curve(precision, recall, auc=metrics["PR_AUC"][i], style=style)
            fig.update_layout(title=target)
            safe_write_image(fig, save_path / "precision_recall_curve" / f"{i:02}. {target} Precision-Recall Curve.png")
