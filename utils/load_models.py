"""Contains functions for loading models."""

import contextlib
from pathlib import Path

import lightning as L  # noqa: N812
import torch
import xgboost as xgb
from omegaconf import OmegaConf

from nn import MODEL_REGISTRY
from nn.fingerprint_creator import FingerprintCreator


def is_lightning_chekpoint(path: str | Path) -> bool:
    """Check if the given path is a PyTorch Lightning checkpoint.

    Parameters
    ----------
    path : str | Path
        Path to the file to be checked.

    Returns
    -------
    bool
        True if the path is a PyTorch Lightning checkpoint, False otherwise.

    """
    if isinstance(path, str):
        path = Path(path)
    return path.suffix == ".ckpt"


class UnknownModelTypeError(ValueError):
    """Unknown model type."""


def load_model(
    model_config: OmegaConf,
) -> torch.nn.Module | L.LightningModule | FingerprintCreator:
    """Load a model from its configuration.

    Parameters
    ----------
    model_config : OmegaConf
        Model configuration.

    Returns
    -------
    torch.nn.Module
        Loaded model.

    Raises
    ------
    UnknownModelTypeError
        If the model type is unknown.

    """
    if model_config.type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_config.type].load_from_config(model_config)
    raise UnknownModelTypeError(model_config.type)


def load_xgboost(xgboost_config: OmegaConf) -> xgb.Booster:
    """Load an XGBoost model from its configuration.

    Parameters
    ----------
    xgboost_config : OmegaConf
        XGBoost model configuration.

    Returns
    -------
    xgb.Booster
        Loaded XGBoost model.

    Notes
    -----
    If the model file does not exist, an empty model is returned.

    """
    xgboost_model = xgb.Booster(xgboost_config.params)
    with contextlib.suppress(KeyError, FileNotFoundError):
        xgboost_model.load_model(xgboost_config.path)

    return xgboost_model
