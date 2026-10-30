"""Contains non-visualisation utility functions."""


def rows_and_cols_from_n_subplots(n_subplots: int) -> tuple[int, int, list[tuple[int, int]]]:
    """COnvert number of subplots to rows and columns.

    Calculate the number of rows and columns required for a figure with n_subplots
    subplots, and return the subplot coordinates.

    Parameters
    ----------
    n_subplots : int
        Number of subplots on the figure.

    Returns
    -------
    rows : int
        Number of rows required for the figure.
    cols : int
        Number of columns required for the figure.
    subplot_coords : list[tuple[int, int]]
        List of subplot coordinates (row, col) for the subplots.

    """
    if n_subplots == 1:
        return 1, 1, [(1, 1)]
    cols = 2
    rows = n_subplots // cols
    if n_subplots % cols != 0:
        rows += 1
    subplot_coords = [(row + 1, col + 1) for row in range(rows) for col in range(cols)]
    return rows, cols, subplot_coords[:n_subplots]


class Corner:
    """Corner positions for subplots."""

    TOP_LEFT = (0.1, 0.9)
    TOP_RIGHT = (0.9, 0.9)
    BOTTOM_LEFT = (0.1, 0.1)
    BOTTOM_RIGHT = (0.9, 0.1)
