"""Validator that compares the uncertainty of the model with the test dataset.

Parameters
----------
dataset : Sequence[DataItem]
    The dataset to evaluate the uncertainty on.
dataset_name : str
    The name of the dataset.

Attributes
----------
validator_init_kwarg_names : ClassVar[frozenset[str]]
    The names of the keyword arguments that can be passed to the validator's
    __init__ method.

"""

import inspect
import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from omegaconf import OmegaConf

from utils.datasets import load_dataset
from utils.filter_kwargs import filter_kwargs
from utils.typing import DataItem
from validation.uncertainty.abstract import UncertaintyValidator

logger = logging.getLogger(__name__)

INVALID_INIT_KWARG_MSG = "Invalid keyword argument '{key}' for {class_name} initialization."


class CompareWithTestValidator(UncertaintyValidator):
    """Validator that compares the uncertainty of the model with the test dataset.

    Parameters
    ----------
    dataset : Sequence[DataItem]
        The dataset to evaluate the uncertainty on.
    dataset_name : str
        The name of the dataset.

    Attributes
    ----------
    validator_init_kwarg_names : ClassVar[frozenset[str]]
        The names of the keyword arguments that can be passed to the validator's
        __init__ method.

    """

    validator_init_kwarg_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        dataset: Sequence[DataItem],
        dataset_name: str,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initializes a CompareWithTestValidator object.

        Parameters
        ----------
        dataset : Sequence[DataItem]
            The dataset to evaluate the uncertainty on.
        dataset_name : str
            The name of the dataset.
        **kwargs : dict[str, Any]
            Additional keyword arguments to be passed to the validator.

        """
        self.validate_validator_init_kwargs(kwargs)
        self.validator_init_kwargs = kwargs
        self.reference_validator = self.validator_class(
            dataset,
            dataset_name,
            **self.validator_init_kwargs,
        )
        self.dataset = self.reference_validator.dataset
        self.dataset_name = self.reference_validator.dataset_name
        self.anchor_validator = None

    @classmethod
    def validate_validator_init_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Validate that the given keyword arguments are valid for initializing the validator.

        Parameters
        ----------
        kwargs : dict[str, Any]
            The keyword arguments to validate.

        Raises
        ------
        TypeError
            If any of the keyword arguments are invalid.

        """
        for key in kwargs:
            if key not in cls.validator_init_kwarg_names:
                raise TypeError(
                    INVALID_INIT_KWARG_MSG.format(key=key, class_name=cls.validator_class.__name__),
                )

    def define_anchor_validator(self, model_config: OmegaConf) -> None:
        """Define the anchor validator.

        Define the anchor validator by loading the test split of the training dataset
        and initializing the validator with the same keyword arguments as the reference
        validator.

        Parameters
        ----------
        model_config : OmegaConf
            The configuration of the model to use.

        """
        dataset_full = load_dataset(
            model_config.training_dataset_name,
        )
        _, __, test = dataset_full.split()
        self.anchor_validator = self.validator_class(
            test,
            f"test_{model_config.training_dataset_name}",
            **self.validator_init_kwargs,
        )

    @abstractmethod
    def run_comparison(
        self,
        model_config: OmegaConf,
        anchor_results: dict,
        reference_results: dict,
        **kwargs: dict,
    ) -> dict:
        """Runs the comparison of the anchor and reference validators.

        Parameters
        ----------
        model_config : OmegaConf
            Model configuration.
        anchor_results : dict
            Results of the anchor validator.
        reference_results : dict
            Results of the reference validator.
        **kwargs : dict
            Additional keyword arguments to be passed to the validators.

        Returns
        -------
        dict
            The results of the comparison.

        """
        raise NotImplementedError

    def run_validation(self, model_config: OmegaConf, **kwargs: dict) -> dict:
        """Runs the comparison of the anchor and reference validators.

        Parameters
        ----------
        model_config : OmegaConf
            Model configuration.
        **kwargs : dict
            Additional keyword arguments to be passed to the validators.

        Returns
        -------
        dict
            The results of the comparison.

        """
        self.define_anchor_validator(model_config)
        logger.info("Running anchor validator")
        anchor_results = self.anchor_validator.validate(
            model_config,
            **filter_kwargs(self.anchor_validator.validate, kwargs),
        )
        logger.info("Running reference validator")
        reference_results = self.reference_validator.validate(
            model_config,
            **filter_kwargs(self.reference_validator.validate, kwargs),
        )
        return self.run_comparison(model_config, anchor_results, reference_results)

    def __init_subclass__(cls, validator_class: type[UncertaintyValidator]) -> None:
        """Initializes a subclass of CompareWithTestValidator.

        Parameters
        ----------
        cls : type
            The subclass to initialize.
        validator_class : type
            The class of the validator to use.

        Notes
        -----
        This method is used to initialize the validator class and the names of the
        keyword arguments that can be passed to the validator. The names of the
        keyword arguments are determined by inspecting the signature of the validator
        class's __init__ method.

        """
        super().__init_subclass__()
        cls.validator_class = validator_class
        init_signature = inspect.signature(cls.validator_class.__init__)
        all_init_kwarg_names = set(init_signature.parameters.keys())
        cls.validator_init_kwarg_names = frozenset(
            all_init_kwarg_names - {"self", "dataset", "dataset_name"},
        )
