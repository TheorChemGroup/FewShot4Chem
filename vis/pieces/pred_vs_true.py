"""Utilities for plotting predicted vs true graphs.

Like this:
     ^
     │    ⁛╱⁛
     │   ⁛╱⁛⁛
pred │  ⁛╱⁛⁛⁛
     │ ⁛╱⁛⁛
     │⁛╱
     └───────────>
          true
"""  # noqa:RUF002

from copy import deepcopy

import numpy as np
import plotly.graph_objects as go

from vis.defaults import COLORS, DEFAULT_STYLE


def plot_pred_vs_true_and_metric(  # noqa: PLR0913 # IMPROVE: split arguments in groups
    pred: np.ndarray,
    true: np.ndarray,
    fig: go.Figure,
    metric_value: float | np.ndarray | np.floating,
    metric_name: str,
    title: str,
    row: int = 1,
    col: int = 1,
    style: dict = DEFAULT_STYLE,
) -> go.Figure:
    """Plot predicted vs true values and a metric.

    Adds a scatter plot of predicted vs true values to the given figure at the given row and col,
    and adds a line y=x, and a text with the value of a given metric.

    Args:
        pred: The predicted values.
        true: The true values.
        fig: The figure to add the scatter plot and line to.
        metric_value: The value of the metric to display.
        metric_name: The name of the metric to display.
        title: The title to display above the subplot.
        row: The row to add the scatter plot and line to. Defaults to 1.
        col: The column to add the scatter plot and line to. Defaults to 1.
        style: The style of the figure.

    Returns:
        The figure with the added scatter plot, line, and text.

    """
    fig.add_trace(
        go.Scatter(
            x=true,
            y=pred,
            mode="markers",
            marker=go.scatter.Marker(color=COLORS["points"], opacity=0.7, size=10),
        ),
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text="True Values", row=row, col=col)
    fig.update_yaxes(title_text="Predicted Values", row=row, col=col)

    x_max = true.max()
    x_min = true.min()
    y_max = pred.max()
    y_min = pred.min()

    min_ = min(x_min, y_min)
    max_ = max(x_max, y_max)
    delta = max_ - min_
    mean = (min_ + max_) / 2
    low_border = mean - delta * 1.1 / 2
    high_border = mean + delta * 1.1 / 2
    xs = np.linspace(low_border, high_border, 100)

    # IMPROVE: move to styles

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=xs,
            mode="lines",
            line=go.scatter.Line(color=COLORS["x=y"], dash="dash"),
            name="x=y",
        ),
        row=row,
        col=col,
    )

    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"{metric_name} = {metric_value:.2f}",
        showarrow=False,
        font={"size": 15, "color": "black"},
        xanchor="left",
        yanchor="top",
    )

    fig.add_annotation(
        x=0.5,
        y=1,
        xref="paper",
        yref="paper",
        text=title,
        showarrow=False,
        font={"size": 18, "color": "black"},
        xanchor="center",
        yanchor="bottom",
    )

    style = deepcopy(style)
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

    style.update(
        {
            "showlegend": False,
            "xaxis_rangemode": "tozero",
            "yaxis_rangemode": "tozero",
            "xaxis_range": [low_border, high_border],
            "yaxis_range": [low_border, high_border],
            "yaxis_scaleanchor": "x",
            "yaxis_scaleratio": 1,
        },
    )

    fig.update_layout(
        **style,
    )

    return fig
