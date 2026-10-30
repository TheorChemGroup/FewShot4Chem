"""Abstract uncertainty validator."""

import logging
from abc import abstractmethod

from omegaconf import OmegaConf

from utils.filter_kwargs import filter_kwargs
from validation.abstract import Validator

logger = logging.getLogger(__name__)


class UncertaintyValidator(Validator):
    """Abstract uncertainty validator."""

    @abstractmethod
    def run_validation(self, model_config: OmegaConf, **kwargs: dict) -> dict:
        """Run validation."""
        raise NotImplementedError

    def save_results(self, results: dict, model_config: OmegaConf, **kwargs: dict) -> None:
        """Save results."""
        raise NotImplementedError

    @abstractmethod
    def plot_results(self, results: dict, model_config: OmegaConf, **kwargs: dict) -> None:
        """Plot results."""
        raise NotImplementedError

    def validate(self, model_config: OmegaConf, **kwargs: dict) -> None:
        """Validate model on dataset by calculating uncertainty metrics.

        Args:
            model_config (OmegaConf): Model configuration.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            None.

        """
        results = self.run_validation(
            model_config=model_config,
            **filter_kwargs(self.run_validation, kwargs),
        )
        self.save_results(
            results=results,
            model_config=model_config,
            **filter_kwargs(self.save_results, kwargs),
        )
        self.plot_results(
            results=results,
            model_config=model_config,
            **filter_kwargs(self.plot_results, kwargs),
        )
        logger.info("Done.")
        return results
