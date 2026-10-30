"""Plot the training results of a model."""

import logging
import re
from pathlib import Path

from plotly.subplots import make_subplots
from tqdm.auto import tqdm

from utils.minor import safe_write_image
from vis.utils import rows_and_cols_from_n_subplots

logger = logging.getLogger(__name__)


def plot_train_results(
    model_name: str,
    n_graphs_on_fig: int = 4,
    figure_width: int = 1400,
    figure_height: int = 1000,
) -> None:
    """Plot the training results of a model.

    Parameters
    ----------
    model_name : str
        The name of the model.
    n_graphs_on_fig : int, optional
        The number of graphs on each figure. Defaults to 4.
    figure_width : int, optional
        The width of the figure. Defaults to 1400.
    figure_height : int, optional
        The height of the figure. Defaults to 1000.

    Notes
    -----
    The function will save the training plots to
    'results/training/<model_name>/version_<last_version>/plots/<i>.png'.

    """
    last_version = -1
    while (Path("logs") / model_name / "lightning_logs" / f"version_{last_version + 1}").exists():
        last_version += 1

    if last_version == -1:
        logger.info("No training logs found")
        return

    with (
        Path("logs")
        / model_name
        / "csv"
        / "lightning_logs"
        / f"version_{last_version}"
        / "metrics.csv"
    ).open() as f:
        titles = f.readline().strip().split(",")

        step_idx = titles.index("step")

        titles.pop(step_idx)

        values = {title: [] for title in titles}

        while line := f.readline():
            read_values = line.strip().split(",")
            step = int(read_values[step_idx])
            read_values.pop(step_idx)
            for title, value in zip(titles, read_values, strict=False):
                values[title].append(
                    (
                        step,
                        (float(value) if re.search(r"\.", value) else int(value))
                        if value
                        else value,
                    ),
                )

    for i in tqdm(range(0, len(titles), n_graphs_on_fig), desc="Generating training result plots"):
        rows, cols, subplot_coords = rows_and_cols_from_n_subplots(
            len(titles[i : i + n_graphs_on_fig]),
        )

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles[i : i + n_graphs_on_fig],
        )
        # Add scatter plots
        for (row_, col_), value in zip(
            subplot_coords,
            [values[title] for title in titles[i : i + n_graphs_on_fig]],
            strict=False,
        ):
            fig.add_scatter(
                x=[pair[0] for pair in value],
                y=[pair[1] for pair in value],
                row=row_,
                col=col_,
                connectgaps=True,
            )

        fig.update_layout(height=figure_height, width=figure_width)
        plots_folder = (
            Path("results") / "training" / model_name / f"version_{last_version}" / "plots"
        )

        safe_write_image(fig, plots_folder / f"{i}.png")

    logger.info("Training plots saved to %s", plots_folder)
