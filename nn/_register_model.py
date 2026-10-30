"""Provides a registry for PyTorch Lightning model types."""

import lightning as L  # noqa: N812
from omegaconf import OmegaConf

MODEL_REGISTRY = {}


def load_from_config_default(cls: type[L.LightningModule], config: OmegaConf) -> L.LightningModule:
    """Loads a PyTorch Lightning model from a configuration."""
    if "path" in config and config.path is not None:
        return cls.load_from_checkpoint(checkpoint_path=config.path, **config.hyperparams)
    return cls(**config.hyperparams)


def register_model_type(name: str):
    """Registers a PyTorch Lightning model type under the given name.

    Parameters
    ----------
    name : str
        The name under which the model type should be registered.

    Returns
    -------
    callable
        A decorator function that registers the model type under the given name.

    """

    def decorator(cls: type[L.LightningModule]) -> type[L.LightningModule]:
        MODEL_REGISTRY[name] = cls
        if not hasattr(cls, "load_from_config"):
            cls.load_from_config = classmethod(load_from_config_default)
        return cls

    return decorator
