"""Contains datasets and utilities for loading datasets."""

# TODO: add dataset classes

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import torch_geometric

from utils._no_filtered_datasets import (
    CLASSES_NAMES,
    DATASET_MAPPING,
    DATASET_OBJECTIVES,
    DATASET_PURPOSES,
    IS_TORCH_GEOMETRIC_DATASET,
    PRE_FILTER_MAPPING,
    TARGETS_DESCRIPTIONS,
    TARGETS_NAMES,
    TRANSFORM_MAPPING,
    VALIDATION_DATASETS,
    get_filtered_targets_descriptions,
    get_filtered_targets_names,
)
from utils.dataset_filter import FilteredDataset

logger = logging.getLogger(__name__)

for key in VALIDATION_DATASETS.copy():

    class F(FilteredDataset, parent_dataset_name=key):  # noqa:D101
        pass

    F.__doc__ = f"""Filtered version of {key} dataset.
    About {key} dataset:
    {DATASET_MAPPING[key].__doc__}"""

    DATASET_MAPPING["filtered_" + key] = F

    TRANSFORM_MAPPING["filtered_" + key] = lambda x: x
    PRE_FILTER_MAPPING["filtered_" + key] = lambda _: True

    TARGETS_DESCRIPTIONS["filtered_" + key] = TARGETS_DESCRIPTIONS[key]
    TARGETS_NAMES["filtered_" + key] = TARGETS_NAMES[key]
    DATASET_OBJECTIVES["filtered_" + key] = DATASET_OBJECTIVES[key]
    if DATASET_OBJECTIVES[key] == "classification":
        CLASSES_NAMES["filtered_" + key] = CLASSES_NAMES[key]
    VALIDATION_DATASETS.add("filtered_" + key)
    DATASET_PURPOSES["filtered_" + key] = DATASET_PURPOSES[key]
    IS_TORCH_GEOMETRIC_DATASET["filtered_" + key] = IS_TORCH_GEOMETRIC_DATASET[key]

AVAILIABLE_DATASETS = list(DATASET_MAPPING.keys())

SPLIT_NAMES = ["train", "val", "test"]

DATASET_NOT_FOUND_MSG = "Dataset with name {} was not found."


def load_dataset(
    dataset_name: str,
    root: str | Path = Path("datasets"),
) -> torch_geometric.data.Dataset:
    """Load a dataset from its name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to be loaded.
    root : str | Path, optional
        Root directory where the dataset is located, by default Path("datasets").

    Returns
    -------
    torch_geometric.data.Dataset
        Loaded dataset.

    Raises
    ------
    KeyError
        If the dataset is not found in DATASET_MAPPING.

    Notes
    -----
    The dataset is loaded with the corresponding transform and pre_filter from
    TRANSFORM_MAPPING and PRE_FILTER_MAPPING, respectively. The dataset's objective
    is set to the corresponding value from DATASET_OBJECTIVES.

    """
    if isinstance(root, str):
        root = Path(root)
    should_split = False
    split_id = None
    try:
        dataset_class = DATASET_MAPPING[dataset_name]
    except KeyError:
        try:
            for split_name in SPLIT_NAMES:
                if dataset_name.startswith(f"{split_name}_"):
                    dataset_name = dataset_name[len(split_name) + 1 :]
                    dataset_class = DATASET_MAPPING[dataset_name]
                    should_split = True
                    split_id = SPLIT_NAMES.index(split_name)
                    break
            else:
                raise
        except KeyError as key_error:
            raise KeyError(DATASET_NOT_FOUND_MSG.format(dataset_name)) from key_error
    if IS_TORCH_GEOMETRIC_DATASET[dataset_name]:
        transform = TRANSFORM_MAPPING[dataset_name]
        pre_filter = PRE_FILTER_MAPPING[dataset_name]
        dataset = dataset_class(
            root=root / dataset_name,
            transform=transform,
            pre_filter=pre_filter,
        )
    else:
        dataset = dataset_class(root=root / dataset_name)
    if dataset_name in DATASET_OBJECTIVES:
        dataset.objective = DATASET_OBJECTIVES[dataset_name]
    dataset.external_name = dataset_name
    if should_split:
        splits = dataset.split()
        dataset = splits[split_id]
    return dataset


class MergeType(Enum):
    """Detrmines how do we merge targets from dataset."""

    STACK = "stack"
    CONCATENATE = "concatenate"


class UnknownMergeTypeError(ValueError):  # IMPROVE: move to errors module
    """Unknown merge type."""


def maybe_load_targets(
    dataset: torch_geometric.data.Dataset,
    merge_type: MergeType = MergeType.STACK,
) -> np.ndarray:
    """Load targets from a dataset.

    If the dataset has a root property, look for a targets.npy file at
    dataset.root / "targets.npy". If it exists, load the targets from there.
    If it does not, load targets from the dataset itself.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        The dataset from which to load the targets.
    merge_type : MergeType, optional
        How to merge the targets. If MergeType.STACK, stack the targets along
        the first axis. If MergeType.CONCATENATE, concatenate the targets along
        the first axis. Defaults to MergeType.STACK.

    Returns
    -------
    targets : numpy.ndarray
        The targets loaded from the dataset.

    Raises
    ------
    UnknownMergeTypeError
        If the merge type is unknown.

    """
    path = None
    if (
        isinstance(dataset, torch_geometric.data.Dataset)
        and dataset.root != torch_geometric.data.dataset.MISSING
    ):
        path = Path(dataset.root) / "targets.npy"
        try:
            res = np.load(path)
            logger.info("Targets succesfully found at %s", path)
        except FileNotFoundError:
            pass
        else:
            return res
    if merge_type == MergeType.CONCATENATE:
        res = np.concatenate([data.y.cpu().numpy() for data in dataset], axis=0)
    elif merge_type == MergeType.STACK:
        res = np.stack([data.y.cpu().numpy() for data in dataset], axis=0)
    else:
        raise UnknownMergeTypeError(merge_type)
    if path is not None:
        np.save(path, res)
    return res
