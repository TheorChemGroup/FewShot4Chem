"""Utility functions for validation."""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch_geometric

from utils.typing import DataItem

logger = logging.getLogger(__name__)


def to_multitarget_metric(
    metric_function: Callable[[np.ndarray, np.ndarray], np.floating],
    *,
    uses_probabilities: bool = False,
    return_list: bool = False,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Convert a metric function for binary classification into one for multiclass classification.

    Parameters
    ----------
    metric_function : Callable[[np.ndarray, np.ndarray], np.floating]
        Metric function for binary classification.
    uses_probabilities : bool, optional
        Whether the metric function uses probabilities or not. If True, the metric
        function expects the second input to be probabilities. If False, the metric
        function expects the second input to be class labels. Defaults to False.
    return_list : bool, optional
        Whether to return a list of metrics or a single metric. Defaults to False.

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        Metric function for multiclass classification.

    """
    if uses_probabilities:
        if return_list:
            return lambda y_true, y_score: [
                metric_function(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])
            ]

        return lambda y_true, y_score: np.array(
            [metric_function(y_true[:, i], y_score[:, i]) for i in range(y_true.shape[1])],
        )

    if return_list:
        return lambda y_true, y_pred: [
            metric_function(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
        ]

    return lambda y_true, y_pred: np.array(
        [metric_function(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])],
    )


def calculate_sample_weights(
    ys: np.ndarray,
    stability_factor: float = 10,
    smoothing: float = 1,
) -> np.ndarray:
    """Calculate sample weights for a classification dataset.

    Parameters
    ----------
    ys : np.ndarray
        Class labels.
    stability_factor : float, optional
        Stability factor. Defaults to 10.
        Decrease if sample weights are too large, increase if they are too small.
    smoothing : float, optional
        Smoothing factor. Defaults to 2.
        The higher the value, the more smooth the sample weights are.

    Returns
    -------
    np.ndarray
        Sample weights for each class.

    """
    logger.debug("ys.shape: %s", ys.shape)
    res = np.zeros_like(ys, dtype=np.float32)
    for i in range(ys.shape[1]):
        class_counts = np.unique(ys[:, i], return_counts=True)[1]
        class_weights = class_counts / class_counts.sum()
        class_weights_smoothed = class_weights ** (1 / (smoothing * ys.shape[1] ** 0.3))
        inverse_class_counts_smoothed = 1 / class_weights_smoothed
        sample_weights_for_target = (
            inverse_class_counts_smoothed * stability_factor / inverse_class_counts_smoothed.sum()
        )
        for class_, weight in enumerate(sample_weights_for_target):
            res[ys[:, i] == class_, i] = weight
    return np.prod(res, axis=1)


def recursively_listify(obj: Any, *, convert_numbers_to_python_types: bool = False) -> Any:  # noqa: ANN401
    # (may accept all types, so `Any` is allowed)
    """Recursively convert any object to a list.

    Parameters
    ----------
    obj : Any
        The object to be converted.
    convert_numbers_to_python_types : bool, optional
        If True, convert numpy numbers to Python types, by default False

    Returns
    -------
    list
        The listified object

    """
    try:
        return [
            recursively_listify(o, convert_numbers_to_python_types=convert_numbers_to_python_types)
            for o in obj
        ]
    except TypeError:
        if convert_numbers_to_python_types:
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
        return obj


def get_collate(dataset: Sequence[DataItem]) -> Callable[[list[DataItem]], Any]:
    """Get the collate function for a dataset.

    If the dataset has a `collate_fn` attribute, return it. Otherwise, if the dataset
    is a sequence of `torch_geometric.data.Data` objects, return
    `torch_geometric.data.Batch.from_data_list`. Otherwise, return
    `torch.utils.data.dataloader.default_collate`.

    Parameters
    ----------
    dataset : Any
        The dataset to get the collate function for.

    Returns
    -------
    Callable[[list], Any]
        The collate function for the dataset.

    """
    try:
        return dataset.collate_fn
    except AttributeError:
        if isinstance(dataset[0], torch_geometric.data.Data):
            return torch_geometric.data.Batch.from_data_list

        return torch.utils.data.dataloader.default_collate
