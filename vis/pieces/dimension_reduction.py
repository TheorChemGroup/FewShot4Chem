"""Contains functions for visualizing the results of dimensionality reduction."""

from copy import deepcopy
from typing import Any

import numpy as np
from plotly import graph_objects as go
from plotly.colors import sample_colorscale

from utils.typing import HasFitTransform
from vis.defaults import COLORS, COLORSCALES, DEFAULT_STYLE


def plot_pred_vs_true_dimension_reduction_results(  # noqa: PLR0913 # IMPROVE: split arguments in groups
    targets: np.ndarray,
    predictions: np.ndarray,
    dimension_reduction_model: HasFitTransform,
    dimension_reduction_model_name: str,
    *,
    fig: go.Figure | None = None,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> go.Figure:
    """Plot the results of the dimensionality reduction.

    Args:
        targets (np.ndarray): The true values.
        predictions (np.ndarray): The predicted values.
        dimension_reduction_model (HasFitTransform): The model used for dimensionality reduction.
        dimension_reduction_model_name (str): The name of the dimensionality reduction model.
        fig (go.Figure, optional): The figure to add the scatter plot to.
        Defaults to None (creates a new figure).
        style (dict[str, Any], optional): The style of the figure. Defaults to DEFAULT_STYLE.

    Returns:
        go.Figure: The created figure.

    """
    combined_data = np.vstack([targets, predictions])

    combined_2d = dimension_reduction_model.fit_transform(combined_data)

    component_1, component_2 = combined_2d[:, 0], combined_2d[:, 1]

    if fig is None:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=component_1[: len(targets)],
            y=component_2[: len(targets)],
            marker_color=COLORS["target"],
            name="targets",
            mode="markers",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=component_1[len(targets) :],
            y=component_2[len(targets) :],
            marker_color=COLORS["predictions"],
            name="predictions",
            mode="markers",
        ),
    )
    style = deepcopy(style)
    style.update(
        {
            "title": f"{dimension_reduction_model_name}-reduced Embeddings",
            "showlegend": True,
        },
    )

    fig.update_layout(**style)
    return fig


def plot_classification_results_with_dimension_reduction(  # noqa: PLR0913 # IMPROVE: split arguments in groups
    features: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    dimension_reduction_model: HasFitTransform,
    dimension_reduction_model_name: str,
    *,
    fig: go.Figure | None = None,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> go.Figure:
    """Plot classification results with dimensionality reduction.

    Args:
        features (np.ndarray): Features to reduce.
        targets (np.ndarray): Targets.
        predictions (np.ndarray): Predictions.
        dimension_reduction_model (HasFitTransform): Model used for dimensionality reduction.
        dimension_reduction_model_name (str): Name of the dimensionality reduction model.
        fig (Optional[go.Figure], optional): The figure to add the scatter plot to.
        Defaults to None (creates a new figure).
        style (dict[str, Any], optional): The style of the figure. Defaults to DEFAULT_STYLE.

    Returns:
        go.Figure: The created figure.

    """
    if fig is None:
        fig = go.Figure()

    features_2d = dimension_reduction_model.fit_transform(features)
    component_1, component_2 = features_2d[:, 0], features_2d[:, 1]

    max_class = int(max(targets.max(), predictions.max()))
    min_class = int(min(targets.min(), predictions.min()))

    classes_colors = dict(
        enumerate(
            sample_colorscale(
                colorscale=COLORSCALES["classes"],
                samplepoints=range(min_class, max_class + 1),
                low=min_class,
                high=max_class,
            ),
            start=min_class,
        ),
    )

    for true_class in range(min_class, max_class + 1):
        properly_predicted = {i for i, true in enumerate(targets) if true == true_class}
        for pred_class in range(min_class, max_class + 1):
            confused = {i for i in properly_predicted if predictions[i] == pred_class}
            fig.add_trace(
                go.Scatter(
                    x=[component_1[i] for i in confused],
                    y=[component_2[i] for i in confused],
                    mode="markers",
                    name=f"True:{true_class} Pred:{pred_class}",
                    opacity=0.6,
                    marker=go.scatter.Marker(
                        color=classes_colors[pred_class],
                        line={
                            "color": classes_colors[true_class],
                            "width": 3,
                        },
                        size=10,
                    ),
                ),
            )
    style = deepcopy(style)
    style.update(
        {
            "title": f"{dimension_reduction_model_name}-reduced Embeddings",
        },
    )

    fig.update_layout(**style)
    return fig
