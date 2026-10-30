"""Contains the class to run a sequence of validators."""

from omegaconf import OmegaConf

from validation.abstract import Validator


class ValidatorSequence(Validator):
    """A sequence of validators to be run in order.

    Parameters
    ----------
    validators : list[Validator]
        A list of validators to be run in sequence.

    Methods
    -------
    validate(model_config: OmegaConf, **kwargs: dict)
        Run the sequence of validators in order.

    """

    def __init__(self, validators: list[Validator]) -> None:
        """Initialize a ValidatorSequence object.

        Args:
            validators : list[Validator]
                A list of validators to be run in sequence.

        """
        self.validators = validators

    def validate(self, model_config: OmegaConf, **kwargs: dict) -> None:
        """Run the sequence of validators in order.

        Args:
            model_config : OmegaConf
                The configuration of the model to be validated.
            **kwargs : dict
                Additional keyword arguments to be passed to the validators.

        """
        for validator in self.validators:
            validator.validate(model_config, **kwargs)
