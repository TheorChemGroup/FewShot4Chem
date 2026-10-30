"""Tests for QM9 target_indices subsetting."""

import numpy as np

# QM9 indices verified on HPC (2026-05-03):
# 2=ε_homo, 3=ε_lumo, 8=U_298
HOMO_IDX = 2
LUMO_IDX = 3
U_IDX = 8


def test_filtered_targets_names_propagates_through_validator() -> None:
    from utils.datasets import TARGETS_NAMES, get_filtered_targets_names

    full = TARGETS_NAMES["QM9"]
    filtered = get_filtered_targets_names("QM9", target_indices=[HOMO_IDX, LUMO_IDX, U_IDX])
    assert len(filtered) == 3
    assert filtered[0] == full[HOMO_IDX]
    assert filtered[1] == full[LUMO_IDX]
    assert filtered[2] == full[U_IDX]


def test_filtered_targets_descriptions_propagates_through_validator() -> None:
    from utils.datasets import TARGETS_DESCRIPTIONS, get_filtered_targets_descriptions

    full = TARGETS_DESCRIPTIONS["QM9"]
    filtered = get_filtered_targets_descriptions(
        "QM9",
        target_indices=[HOMO_IDX, LUMO_IDX, U_IDX],
    )
    assert len(filtered) == 3
    assert filtered[0] == full[HOMO_IDX]


def test_filtered_targets_names_none_returns_full_list() -> None:
    from utils.datasets import TARGETS_NAMES, get_filtered_targets_names

    full = TARGETS_NAMES["QM9"]
    result = get_filtered_targets_names("QM9", target_indices=None)
    assert result == list(full)


def test_maybe_subset_targets_slices_correctly() -> None:
    from validation.target.abstract import TargetValidator

    rng = np.random.default_rng(0)
    targets = rng.standard_normal((50, 16)).astype(np.float32)

    sliced = TargetValidator._maybe_subset_targets(  # noqa: SLF001
        targets,
        target_indices=[HOMO_IDX, LUMO_IDX, U_IDX],
    )
    assert sliced.shape == (50, 3)
    np.testing.assert_array_equal(sliced[:, 0], targets[:, HOMO_IDX])
    np.testing.assert_array_equal(sliced[:, 1], targets[:, LUMO_IDX])
    np.testing.assert_array_equal(sliced[:, 2], targets[:, U_IDX])


def test_maybe_subset_targets_passthrough_when_none() -> None:
    from validation.target.abstract import TargetValidator

    rng = np.random.default_rng(1)
    targets = rng.standard_normal((50, 16)).astype(np.float32)
    result = TargetValidator._maybe_subset_targets(targets, target_indices=None)  # noqa: SLF001
    np.testing.assert_array_equal(result, targets)
