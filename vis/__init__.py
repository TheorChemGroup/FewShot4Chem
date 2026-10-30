"""Contains utility functions for visualizing the results of training models."""

try:  # needs deeplake, which only the atomic-embedding plots use
    from vis.atomic_embeddings import plot_atomic_embeddings
except ImportError:
    plot_atomic_embeddings = None
from vis.pieces import (
    plot_classification_results_with_dimension_reduction,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_pred_vs_true_and_metric,
    plot_pred_vs_true_dimension_reduction_results,
    plot_roc_curve,
)
from vis.train_results import plot_train_results

__all__ = [
    "plot_atomic_embeddings",
    "plot_classification_results_with_dimension_reduction",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_pred_vs_true_and_metric",
    "plot_pred_vs_true_dimension_reduction_results",
    "plot_roc_curve",
    "plot_train_results",
]
