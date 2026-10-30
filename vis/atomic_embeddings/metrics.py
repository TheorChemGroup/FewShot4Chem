"""Plot and save metrics from a list of results dictionaries."""

import logging
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.minor import safe_write_image

logger = logging.getLogger(__name__)


def plot_and_save_metrics(
    results: list[dict[str, Any]],
    dataset_name: str,
    model_name: str,
    figure_width: int = 1400,
    figure_height: int = 1000,  # IMPROVE: move to styles
) -> None:
    # Extract model names
    """Plots and saves metrics from a list of results dictionaries.

    Parameters
    ----------
    results : list[dict[str, Any]]
        A list of dictionaries containing the results of the models.
        Each dictionary should have a "Model" key with the name of the model as its value,
        and other keys with the names of the metrics as their values.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    figure_width : int, optional
        The width of the figure to be plotted. Defaults to 1400.
    figure_height : int, optional
        The height of the figure to be plotted. Defaults to 1000.

    """
    model_names = [res["Model"] for res in results]

    # Extract metric names
    metric_names = list(results[0].keys())
    metric_names.remove("Model")

    # Create subplots grid (4x2)
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=metric_names,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Add traces for each metric
    for i, metric in enumerate(metric_names, 1):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        # Get all model values for this metric
        values = [res[metric] for res in results]

        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                name=metric,
                text=[round(v, 3) for v in values],
                textposition="auto",
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=figure_height,
        width=figure_width,
        title_text="Multiclass Model Comparison",
        showlegend=False,
    )

    # Add axis labels
    fig.update_xaxes(title_text="Model", row=4, col=1)
    fig.update_xaxes(title_text="Model", row=4, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)

    safe_write_image(Path("results") / dataset_name / model_name / "plots" / "metrics.png")
