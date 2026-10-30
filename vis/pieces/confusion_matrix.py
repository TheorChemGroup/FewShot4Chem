"""Contains functions for plotting a confusion matrix."""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import plotly.graph_objects as go

from vis.constants import CLASSES_NAMES_FOR_BINARY_CLASSIFICATION
from vis.defaults import DEFAULT_STYLE


def plot_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    classes: Sequence[str],
    style: dict[str, Any] = DEFAULT_STYLE,
    fig: go.Figure | None = None,
    metrics: dict[str, Any] | None = None,
) -> go.Figure:
    """Plots a confusion matrix.

    Parameters
    ----------
    confusion_matrix : Sequence[Sequence[int]]
        A 2D array containing the confusion matrix
    classes : Sequence[str]
        The class names corresponding will be used to label the x and y axes
    style : dict[str, Any], optional
        A dictionary containing visual style options for the plot
    fig : go.Figure, optional
        A pre-existing plotly figure to add the confusion matrix to
    metrics : dict[str, Any], optional
        A dictionary containing any additional metrics to be displayed
        below the confusion matrix

    Returns
    -------
    go.Figure
        A plotly figure containing the confusion matrix and any additional metrics

    """
    classes = list(classes)
    style = deepcopy(style)
    if fig is None:
        fig = go.Figure()
    if len(classes) == len(CLASSES_NAMES_FOR_BINARY_CLASSIFICATION) and {
        classes[i].lower() for i in range(len(classes))
    } == {
        CLASSES_NAMES_FOR_BINARY_CLASSIFICATION[i].lower()
        for i in range(len(CLASSES_NAMES_FOR_BINARY_CLASSIFICATION))
    }:
        if classes[0].lower() != CLASSES_NAMES_FOR_BINARY_CLASSIFICATION[0].lower():
            confusion_matrix = [
                [confusion_matrix[1][1], confusion_matrix[1][0]],
                [confusion_matrix[0][1], confusion_matrix[0][0]],
            ]
        classes = CLASSES_NAMES_FOR_BINARY_CLASSIFICATION
        annotations = [["TP({})", "FP({})"], ["FN({})", "TN({})"]]
        text = [
            [ann.format(val) for ann, val in zip(row_ann, row_values, strict=True)]
            for row_ann, row_values in zip(annotations, confusion_matrix, strict=True)
        ]
        style["yaxis_autorange"] = "reversed"
    else:
        text = confusion_matrix
    fig.add_trace(
        go.Heatmap(
            z=confusion_matrix,
            x=classes,
            y=classes,
            colorscale="Viridis",
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
        ),
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
    if metrics is not None:
        annotation_spacer = len(metrics) * 20 + 10
        style["height"] += annotation_spacer
        style["margin"] = {"l": 107, "r": 107, "t": 107, "b": 107 + annotation_spacer}
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            xanchor="center",
            yanchor="top",
            text="<br>".join(
                [f"{metric}: {value}" for metric, value in metrics.items()],
            ),
            showarrow=False,
            font={"size": 14},
        )
    style["xaxis_showline"] = False
    style["yaxis_showline"] = False

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        **style,
    )

    return fig
