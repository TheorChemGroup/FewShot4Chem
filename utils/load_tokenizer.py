"""Provides functions for loading tokenizers, their configurations, and trainer parameters."""

from pathlib import Path

from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import BPE


def load_tokenizer_config(tokenizer_name: str) -> dict:
    """Loads tokenizer configuration from its configuration file.

    Parameters
    ----------
    tokenizer_name : str
        Name of the tokenizer.

    Returns
    -------
    dict
        Tokenizer configuration.

    """
    return OmegaConf.load(Path("models") / "configs" / "tokenizers" / f"{tokenizer_name}.yaml")


def load_tokenizer(tokenizer_name: str) -> Tokenizer:
    """Loads a tokenizer from its configuration.

    Parameters
    ----------
    tokenizer_name : str
        Name of the tokenizer.

    Returns
    -------
    Tokenizer
        Loaded tokenizer.

    Notes
    -----
    If the tokenizer has been trained, it is loaded from its save file.
    Otherwise, a new tokenizer is created with the specified configuration.

    """
    tokenizer_config = load_tokenizer_config(tokenizer_name)
    if tokenizer_config.path is not None:
        tokenizer = Tokenizer.from_file(tokenizer_config.path)
        tokenizer.trained = True
    else:
        tokenizer = Tokenizer(BPE(**tokenizer_config.bpe_params))
        tokenizer.trained = False
        if tokenizer_config.get("padding_params") is not None:
            tokenizer.enable_padding(**tokenizer_config.padding_params)
    return tokenizer


def get_trainer_params(tokenizer_name: str) -> dict:
    """Returns trainer parameters for a given tokenizer.

    Parameters
    ----------
    tokenizer_name : str
        Name of the tokenizer.

    Returns
    -------
    dict
        Trainer parameters.

    """
    tokenizer_config = load_tokenizer_config(tokenizer_name)
    return tokenizer_config.trainer_params
