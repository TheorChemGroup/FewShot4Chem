"""Contains functions for drawing some useful plots."""

from vis.pieces.auc_metrics import plot_precision_recall_curve, plot_roc_curve
from vis.pieces.boxplot import plot_boxplot
from vis.pieces.confusion_matrix import plot_confusion_matrix
from vis.pieces.dimension_reduction import (
    plot_classification_results_with_dimension_reduction,
    plot_pred_vs_true_dimension_reduction_results,
)
from vis.pieces.histogram import plot_histogram
from vis.pieces.pred_vs_true import plot_pred_vs_true_and_metric

__all__ = [
    "plot_boxplot",
    "plot_classification_results_with_dimension_reduction",
    "plot_confusion_matrix",
    "plot_histogram",
    "plot_precision_recall_curve",
    "plot_pred_vs_true_and_metric",
    "plot_pred_vs_true_dimension_reduction_results",
    "plot_roc_curve",
]
