"""Plots representing AUC metrics, PR and ROC curves."""

from copy import deepcopy
from typing import Any

import numpy as np
from plotly import graph_objects as go

from vis.defaults import COLORS, DEFAULT_STYLE
from vis.utils import Corner


def plot_curve_with_area(  # noqa: PLR0913 # TODO: split arguments in groups
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    line_color: str,  # IMPROVE: move to style
    fillcolor: str,  # IMPROVE: move to style
    *,
    name: str | None = None,
    metric_info_str: str | None = None,
    info_corner: Corner = Corner.TOP_LEFT,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> go.Figure:
    """Plot a curve with an area under it, with a given title, and optional extra information.

    Parameters
    ----------
    xs : np.ndarray
        The x values of the curve.
    ys : np.ndarray
        The y values of the curve.
    title : str
        The title of the plot.
    name : Optional[str], default: None
        The name of the curve. If None, no legend is shown.
    metric_info_str : Optional[str], default: None
        A string with extra information about the metric.
        If None, no extra information is shown.
    info_corner : Corner, default: Corner.TOP_LEFT
        The corner of the plot to show the extra information.
    line_color : str
        The color of the curve.
    fillcolor : str
        The color of the area under the curve.
    style : dict[str, Any], default: DEFAULT_STYLE
        The style of the plot.

    Returns
    -------
    fig : go.Figure
        The created figure.

    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            fill="tozeroy",
            name=name,
            fillcolor=fillcolor,
            line_color=line_color,
            opacity=0.3,
        ),
    )
    fig.add_annotation(
        text=metric_info_str,
        xref="paper",
        yref="paper",
        x=info_corner[0],
        y=info_corner[1],
        showarrow=False,
    )
    style = deepcopy(style)
    style.update(
        {
            "title": title,
            "showlegend": name is not None,
            "xaxis_range": [0, 1],
            "yaxis_range": [0, 1],
            "xaxis_rangemode": "tozero",
            "yaxis_rangemode": "tozero",
            "margin_autoexpand": False,
        },
    )
    if "width" in style and "height" not in style:
        style["height"] = style["width"]
    elif "height" in style and "width" not in style:
        style["width"] = style["height"]
    elif "width" in style and "height" in style:
        style["width"] = min(style["width"], style["height"])
        style["height"] = style["width"]
    else:
        style["width"] = 800
        style["height"] = 800

    fig.update_layout(
        **style,
    )
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    *,
    info_corner: Corner = Corner.BOTTOM_RIGHT,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> go.Figure:
    """Plot the ROC curve.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    auc : float
        Area under the curve.
    info_corner : Corner, optional
        Where to put the text with the AUC value.
        Defaults to Corner.BOTTOM_RIGHT.
    style : dict[str, Any], optional
        The style of the plot.
        Defaults to DEFAULT_STYLE.

    Returns
    -------
    fig : go.Figure
        The created figure.

    """
    return plot_curve_with_area(
        xs=fpr,
        ys=tpr,
        title="ROC Curve",
        line_color=COLORS["roc_curve"],
        fillcolor=COLORS["roc_auc"],
        name=None,
        metric_info_str=f"ROC AUC: {auc:.3f}",
        info_corner=info_corner,
        style=style,
    )


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    auc: float,
    *,
    info_corner: Corner = Corner.BOTTOM_RIGHT,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> go.Figure:
    """Plot the PR curve.

    Parameters
    ----------
    precision : np.ndarray
        Precision values.
    recall : np.ndarray
        Recall values.
    auc : float
        Area under the curve.
    info_corner : Corner, optional
        Where to put the text with the AUC value.
        Defaults to Corner.BOTTOM_RIGHT.
    style : dict[str, Any], optional
        The style of the plot.
        Defaults to DEFAULT_STYLE.

    Returns
    -------
    fig : go.Figure
        The created figure.

    """
    return plot_curve_with_area(
        xs=recall,
        ys=precision,
        title="PR Curve",
        name=None,
        line_color=COLORS["pr_curve"],
        fillcolor=COLORS["pr_auc"],
        metric_info_str=f"PR AUC: {auc:.3f}",
        info_corner=info_corner,
        style=style,
    )
