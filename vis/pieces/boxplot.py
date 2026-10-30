"""Contains functions for plotting a boxplot."""

from typing import Any

from plotly import graph_objects as go

from vis.defaults import BOXPLOT_STYLE, DEFAULT_STYLE


def plot_boxplot(
    fig: go.Figure,
    data: list[float],
    name: str,
    *,
    boxplot_style: dict[str, Any] = BOXPLOT_STYLE,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> go.Figure:
    """Plot a boxplot.

    Parameters
    ----------
    fig : go.Figure
        The figure to add the boxplot to.
    data : list[float]
        The data to plot.
    name : str
        The name of the boxplot.
    boxplot_style : dict[str, Any], optional
        The style of the boxplot.
        Defaults to BOXPLOT_STYLE.
    style : dict[str, Any], optional
        The style of the plot.
        Defaults to DEFAULT_STYLE.

    Returns
    -------
    fig : go.Figure
        The created figure.

    """
    fig.add_trace(
        go.Box(
            y=data,
            name=name,
            **boxplot_style,
        ),
    )
    style = style.copy()
    style["title"] = name
    fig.update_layout(**style)
    return fig
