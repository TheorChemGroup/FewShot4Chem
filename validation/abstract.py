"""Super-abstract class for validation is implemented here."""

from abc import ABC, abstractmethod

from omegaconf import OmegaConf


class Validator(ABC):
    """Super-abstract class for validation is implemented here."""

    @abstractmethod
    def validate(self, model_config: OmegaConf, **kwargs: dict) -> None:
        """Run validation."""
        raise NotImplementedError
