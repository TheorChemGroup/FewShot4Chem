"""Select the right validator for a dataset using a head_config."""

import logging

from omegaconf import OmegaConf

from utils.datasets import DATASET_PURPOSES, load_dataset
from validation.abstract import Validator
from validation.target import ClassificationValidator, RegressionValidator
from validation.uncertainty import CompareEGLWithTestValidator, EGLValidator

logger = logging.getLogger(__name__)

UNKNOWN_VALIDATOR_SPECIFICATION_MSG = "Unknown validator specification: {specification}"


class UnknownObjectiveError(ValueError):
    """Unknown objective type."""


def select_validator(
    head_config: OmegaConf,
    dataset_name: str,
    validator_str: str = "boosting",
    random_seed: int = 42,
) -> Validator:
    if validator_str == "boosting":
        return select_proper_boosting_validator(head_config, dataset_name, random_seed=random_seed)
    if validator_str == "uncertainty":
        dataset = load_dataset(dataset_name)
        return CompareEGLWithTestValidator(dataset=dataset, dataset_name=dataset_name)
    if DATASET_PURPOSES[dataset_name] == "training":
        logger.warning(
            "validator_str=%r not recognized; falling back to EGLValidator because "
            "dataset %r has purpose='training'.",
            validator_str,
            dataset_name,
        )
        return EGLValidator(dataset_name=dataset_name)
    raise ValueError(UNKNOWN_VALIDATOR_SPECIFICATION_MSG.format(specification=validator_str))


def select_proper_boosting_validator(
    head_config: OmegaConf,
    dataset_name: str,
    random_seed: int = 42,
) -> Validator:
    dataset = load_dataset(dataset_name)
    if dataset.objective == "classification":
        return ClassificationValidator(
            head_config=head_config, dataset=dataset, random_seed=random_seed,
        )
    if dataset.objective == "regression":
        return RegressionValidator(
            head_config=head_config, dataset=dataset, random_seed=random_seed,
        )
    raise UnknownObjectiveError(dataset.objective)
