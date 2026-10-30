"""Functions to plot boundaries between classes of atomic embeddings."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.svm import SVC

from utils.minor import safe_write_image
from vis.atomic_embeddings.utils import (
    get_svc_boundaries,
    idx_2_atom,
)
from vis.defaults import DEFAULT_STYLE

logger = logging.getLogger(__name__)


NO_N_COMPONENTS_ERROR = "Reducer must have n_components parameter"


def reduce_and_plot_boundaries(  # noqa: PLR0913
    svc: SVC,
    x: np.ndarray,
    y: np.ndarray,
    reducer: type,
    # IMPROVE: change variable names to the ones analagous o the functions from vis.__init__
    save_folder: str,
    boundary_model_name: str,
    reducer_name: str,
    *,
    style: dict[str, Any] = DEFAULT_STYLE,
) -> None:
    # Ensure reducer outputs 2 components
    """Plot the decision boundaries for a given Support Vector Machine model.

    The function takes in the model, the data points to be plotted, and the reducer to be used.
    The reducer should be a class with a fit_transform() method that takes in the data points and
    returns the reduced data points.

    The function also takes in the save folder, the name of the boundary model, the name of the
    reducer, and the style of the plot.

    The function first reduces the data points using the given reducer. It then gets the decision
    boundaries of the given model. The function then plots the reduced data points and the decision
    boundaries in a scatter plot. The plot is then saved to the given save folder with the given
    boundary model name and reducer name.

    Parameters
    ----------
    svc : SVC
        The Support Vector Machine model to be plotted.
    x : np.ndarray
        The data points to be plotted.
    y : np.ndarray
        The labels of the data points.
    reducer : type
        The reducer to be used. Should be a class with a fit_transform() method.
    save_folder : str
        The folder to save the plot in.
    boundary_model_name : str
        The name of the boundary model to be used in the filename.
    reducer_name : str
        The name of the reducer to be used in the filename.
    style : dict[str, Any]
        The style of the plot. defaults to DEFAULT_STYLE (from vis.defaults)

    """
    if hasattr(reducer, "n_components"):
        reducer.n_components = 2
    else:
        raise ValueError(NO_N_COMPONENTS_ERROR)

    boundaries = get_svc_boundaries(svc, x)

    x_reduced = reducer.fit_transform(x)

    # Create Plotly figure
    fig = go.Figure()

    # Add data points
    for label in np.unique(y):
        mask = y == label
        fig.add_trace(
            go.Scatter(
                x=x_reduced[mask, 0],
                y=x_reduced[mask, 1],
                mode="markers",
                name=idx_2_atom[label],
                marker={"size": 8},
                zorder=1,
            ),
        )

    for boundary in boundaries:
        boundary_2d = reducer.transform(boundary["boundary_points"])
        # Add decision regions
        fig.add_trace(
            go.Scatter(
                x=boundary_2d[:, 0],
                y=boundary_2d[:, 1],
                mode="markers",
                name=(
                    f"{idx_2_atom[boundary['classes'][0]]} vs {idx_2_atom[boundary['classes'][1]]}"
                    if boundary["strategy"] == "ovo"
                    else f"{idx_2_atom[boundary['classes'][0]]} vs rest"
                ),
                zorder=2,
            ),
        )

    # Add axis labels and title
    style = deepcopy(style)

    style.update(
        title=f"{boundary_model_name} Decision Boundaries with {reducer_name}",
        xaxis_title=f"{reducer_name} 1",
        yaxis_title=f"{reducer_name} 2",
        legend_title="Classes",
        margin={
            "t": 50,
        },
    )
    fig.update_layout(**style)

    # Save outputs
    safe_write_image(fig, Path(save_folder) / f"{boundary_model_name}_{reducer_name}_boundary.png")
