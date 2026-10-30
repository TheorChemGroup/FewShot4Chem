"""Contains functions for plotting a histogram."""

from collections.abc import Callable
from typing import Any

import numpy as np
from plotly import graph_objects as go
from scipy.stats import norm

from vis.defaults import HISTOGRAM_STYLE


def filter_outliers_by_confidence(data: np.ndarray, confidence: float = 0.95) -> list[float]:
    """Filter outliers from a list of data.

    This function filters out the points that are outside the confidence interval.

    Parameters
    ----------
    data : list[float]
        The data to filter.
    confidence : float
        The confidence level to use for filtering.

    Returns
    -------
    np.ndarray
        The filtered data.

    """
    mean = data.mean()
    std = data.std()
    lower_bound, upper_bound = (
        norm.ppf(1 - confidence) * std / np.sqrt(len(data)) + mean,
        norm.ppf(confidence) * std / np.sqrt(len(data)) + mean,
    )
    return data[(data > lower_bound) & (data < upper_bound)]


def filter_outliers_by_fraction(data: np.ndarray, fraction: float = 0.05) -> list[float]:
    """Filter outliers from a list of data.

    This function filters out the fraction of the most remote from the mean points.

    Args:
        data (np.ndarray): The data to filter.
        fraction (float): The fraction of the most remote points to filter out.

    Returns:
        np.ndarray: The filtered data.

    """
    mean = data.mean()
    argsort = np.argsort(np.abs(data - mean))
    sorted_data = data[argsort]
    return sorted_data[: -int(len(sorted_data) * fraction)]


def plot_histogram(
    fig: go.Figure,
    data: np.ndarray | list[float],
    name: str,
    *,
    style: dict[str, Any] = HISTOGRAM_STYLE,
    histogram_kwargs: dict[str, Any] | None = None,
    filter_outliers: Callable[[np.ndarray], np.ndarray] = filter_outliers_by_fraction,
) -> go.Figure:
    """Plot a histogram.

    Parameters
    ----------
    fig : go.Figure
        The figure to add the histogram to.
    data : list[float]
        The data to plot.
    name : str
        The name of the histogram.
    style : dict[str, Any], optional
        The style of the plot.
        Defaults to HISTOGRAM_STYLE.
    histogram_kwargs : dict[str, Any], optional
        The keyword arguments to pass to the histogram.
    filter_outliers : Callable[[np.ndarray], np.ndarray], optional
        The function to use for filtering outliers.
        Defaults to filter_outliers_by_confidence.

    Returns
    -------
    fig : go.Figure
        The created figure.

    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if histogram_kwargs is None:
        histogram_kwargs = {}
    fig.add_trace(
        go.Histogram(
            x=filter_outliers(data),
            name=name,
            **histogram_kwargs,
        ),
    )
    style = style.copy()
    style["title"] = name
    fig.update_layout(**style)
    return fig
