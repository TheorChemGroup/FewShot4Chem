"""Non-visualisation utility functions."""

import logging
from collections.abc import Callable
from itertools import combinations, pairwise
from typing import Any

import numpy as np
from sklearn.svm import SVC
from sklearn.utils import check_array
from tqdm.auto import tqdm

from utils.constants import ATOM_2_IDX

idx_2_atom = {v: k for k, v in ATOM_2_IDX.items()}

logger = logging.getLogger(__name__)


def golden_ratio_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Golden ratio search for root finding in [a, b]."""
    phi = (1 + np.sqrt(5)) / 2
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    f_c = f(c)
    f_d = f(d)

    for _ in range(max_iter):
        if np.sign(f_c) != np.sign(f_d):
            a, b = c, d
        elif np.sign(f(a)) == np.sign(f_c):
            a = c
        else:
            b = d

        c = b - (b - a) / phi
        d = a + (b - a) / phi
        f_c = f(c)
        f_d = f(d)

        if abs(b - a) < tol:
            break

    return (a + b) / 2


class BoundarySearchParameters:
    """Parameters for boundary search."""

    n_pairs: int
    n_random_samples: int
    tol: float
    max_iter: int
    n_intervals: int

    def __init__(
        self,
        *,
        n_pairs: int = 1000,
        n_random_samples: int = 1000,
        tol: float = 1e-3,
        max_iter: int = 100,
        n_intervals: int = 100,
    ) -> None:
        """Initializes a BoundarySearchParameters object.

        Parameters
        ----------
        n_pairs : int, optional
            Max pairs to sample per class combination, by default 1000
        n_random_samples : int, optional
            Number of random samples, by default 1000
        tol : float, optional
            Tolerance for root finding, by default 1e-3
        max_iter : int, optional
            Max iterations for golden ratio search, by default 100
        n_intervals : int, optional
            Number of interpolation points, by default 100

        """
        self.n_pairs = n_pairs
        self.n_random_samples = n_random_samples
        self.tol = tol
        self.max_iter = max_iter
        self.n_intervals = n_intervals


def get_svc_boundaries(
    svc: SVC,
    x: np.ndarray,
    *,
    search_params: BoundarySearchParameters | None = None,
) -> list[dict[str, Any]]:
    """Find decision boundaries using line interpolation between class points.

    Args:
        svc: Trained SVC model
        x: Training data array
        search_params: Parameters for boundary search. Use a special BoundarySearchParameters class
            for this. Search parameters:
                n_pairs: Max pairs to sample per class combination
                tol: Tolerance for root finding
                max_iter: Max iterations for golden ratio search
                n_intervals: Number of interpolation points
                n_random_samples: Number of random samples

    Returns:
        List of boundary points for each class pair

    """
    if search_params is None:
        search_params = BoundarySearchParameters()
    x = check_array(x)
    min_vals = x.min(axis=0)
    max_vals = x.max(axis=0)
    n_features = x.shape[-1]
    x = np.concatenate(
        (
            x,
            np.random.default_rng().uniform(
                min_vals,
                max_vals,
                size=(search_params.n_random_samples, n_features),
            ),
        ),
    )

    y_pred = svc.predict(x)
    classes = svc.classes_
    is_ovo = svc.decision_function_shape == "ovo"
    boundaries = []
    line = lambda t, a, b: a + t * (b - a)  # noqa: E731

    # Generate class pairs
    class_pairs = list(combinations(classes, 2)) if is_ovo else [(c,) for c in classes]

    for pair_idx, pair in enumerate(class_pairs):
        if is_ovo:
            # OvO case: get points from two classes
            i, j = pair
            mask_i = y_pred == i
            mask_j = y_pred == j
            points_i = x[mask_i]
            points_j = x[mask_j]
            pair_name = f"{idx_2_atom[i]} vs {idx_2_atom[j]}"
        else:
            # OvR case: get points from one class vs others
            i = pair[0]
            mask_i = y_pred == i
            points_i = x[mask_i]
            points_j = x[~mask_i]
            pair_name = f"{idx_2_atom[i]} vs others"

        # Check if we have points from both groups
        if len(points_i) == 0 or len(points_j) == 0:
            continue

        # Sample pairs of points
        n_samples = min(search_params.n_pairs, len(points_i), len(points_j))
        idx_i = np.random.default_rng().choice(len(points_i), n_samples, replace=False)
        idx_j = np.random.default_rng().choice(len(points_j), n_samples, replace=False)

        boundary_points = []

        if logger.isEnabledFor(logging.INFO):
            progress_bar = tqdm(
                zip(points_i[idx_i], points_j[idx_j], strict=False),
                total=len(idx_i),
                desc=(f"Finding boundaries for {pair_name}"),
                unit="pairs",
            )
        else:
            progress_bar = zip(points_i[idx_i], points_j[idx_j], strict=False)

        for a, b in progress_bar:
            # Decision function for this pair
            if is_ovo:
                dec_func = (  # noqa: PLC3002
                    lambda i, a_, b_: (
                        lambda t_: svc.decision_function(line(t_, a_, b_).reshape(1, -1))[
                            :,
                            i,
                        ]
                    )
                )(pair_idx, a, b)
            else:
                dec_func = (  # noqa: PLC3002
                    lambda i, a_, b_: (
                        lambda t_: svc.decision_function(line(t_, a_, b_).reshape(1, -1))[
                            :,
                            i,
                        ]
                    )
                )(pair_idx, a, b)
            # Find root using golden ratio search
            intervals = np.linspace(0, 1, search_params.n_intervals)
            for start, end in pairwise(intervals):
                t = golden_ratio_root(
                    dec_func,
                    start,
                    end,
                    search_params.tol,
                    search_params.max_iter,
                )

                if start <= t <= end:
                    boundary_point = line(t, a, b)
                    boundary_points.append(boundary_point)
                    break

        logger.info(
            "Found %d boundary points for %s",
            len(boundary_points),
            pair_name,
        )

        boundaries.append(
            {
                "classes": pair,
                "boundary_points": np.array(boundary_points),
                "kernel": svc.kernel,
                "strategy": "ovo" if is_ovo else "ovr",
            },
        )

    return boundaries


def _create_full_grid(
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    n_points_per_dim: int,
) -> np.ndarray:
    """Create full grid for 2D case."""
    x = np.linspace(min_vals[0], max_vals[0], n_points_per_dim)
    y = np.linspace(min_vals[1], max_vals[1], n_points_per_dim)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


class NotA2DVectorError(ValueError):
    """Not a 2D vector."""


def _create_hyperplane_grid(
    w: np.ndarray,
    b: float,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Generate points on hyperplane within bounds (for 2D linear case)."""
    if len(w) != 2:  # noqa: PLR2004
        raise NotA2DVectorError(w)

    x_min, x_max = min_vals[0], max_vals[0]
    y_min, y_max = min_vals[1], max_vals[1]

    # Create grid along x-axis
    x = np.linspace(x_min, x_max, n_points)
    y = (-w[0] * x - b) / w[1]

    # Filter points within y bounds
    valid = (y >= y_min) & (y <= y_max)
    return np.column_stack((x[valid], y[valid]))


def recursively_set_values(dict_: dict, value: Any) -> None:  # noqa: ANN401
    """Recursively sets all values in a dictionary to a certain value.

    Args:
        dict_ (dict): The dictionary to be modified.
        value: The value to be set.

    """
    for k, v in dict_.items():
        if isinstance(v, dict):
            recursively_set_values(v, value)
        else:
            dict_[k] = value


def recursively_merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Recursively merges two dictionaries.

    Args:
        dict1 (dict): The dictionary to be modified.
        dict2 (dict): The dictionary with new values.

    Returns:
        dict: The modified dictionary.

    """
    for k, v in dict2.items():
        if isinstance(v, dict):
            if k not in dict1:
                dict1[k] = {}
            dict1[k] = recursively_merge_dicts(dict1[k], v)
        else:
            dict1[k] = v
    return dict1
