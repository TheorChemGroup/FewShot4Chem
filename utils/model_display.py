"""Minor utilities for model display (pretty names, summary)."""

import logging
import re
from pathlib import Path

from lightning.pytorch.utilities.model_summary import ModelSummary, get_human_readable_count
from omegaconf import OmegaConf

from utils.load_models import load_model

logger = logging.getLogger(__name__)

PRETTY_MODEL_TYPES = {"gnn": "GNN", "chemberta": "ChemBERTa"}


class ModelNotFoundError(Exception):
    """Exception raised when a model is not found."""


def search_for_model(model_name: str) -> Path:
    """Searches for a model configuration file in the 'models/configs' directory.

    Args:
        model_name: The name of the model to search for.

    Returns:
        The path to the model configuration file if it exists, otherwise None.

    """
    configs_dir = Path("models") / "configs"
    for type_dir in configs_dir.iterdir():
        for cfg_file in type_dir.iterdir():
            if cfg_file.stem == model_name:
                return cfg_file
    raise ModelNotFoundError


def get_model_info(model_folder: str) -> dict[str, str]:
    """Returns the model info for the given model name.

    Args:
        model_folder: The folder name of the model.

    Returns:
        A dictionary containing the model info.

    """
    model_name = model_folder.name
    try:
        model_path = search_for_model(model_name)
    except ModelNotFoundError:
        logger.warning("Model %s not found", model_name)
        return None
    model_config = OmegaConf.load(model_path)
    return {
        "name": model_name,
        "path": model_path,
        "type": model_config.type,
        "config": model_config,
    }


DATETIME_REGEX = r"\d{2}\.\d{2}\.\d{4}_\d{2}-\d{2}"


def create_pretty_name(
    model_config: OmegaConf,
    model_name: str,
    model_path: Path | None = None,
) -> str | None:
    """Creates a pretty name for the given model folder.

    The pretty name is generated based on the model type and number of parameters.
    If the model is a PyTorch Lightning module, the number of parameters is obtained
    using the ModelSummary class. If the model is a FingerprintCreator, the number of
    dimensions of the Morgan fingerprints is used.

    Args:
        model_config: The model configuration.
        model_name: The name of the model.
        model_path: The path to the model configuration file. Used to save the number of parameters
        for future use. If None, the number of parameters will not be saved.

    Returns:
        The pretty name of the model, or None if the model type is unknown.

    """
    if model_config.type in {"gnn", "chemberta"}:
        date_found = re.search(DATETIME_REGEX, model_name)
        maybe_date = f" ({date_found.group()})" if date_found else ""
        is_dummy = model_config.get("path", None) is None and not (
            model_config.type == "chemberta" and model_config.get("pretrained", True)
        )
        maybe_not_trained = " (not trained)" if is_dummy else ""
        if "n_params" in model_config:
            n_params = model_config.n_params
        else:
            model = load_model(model_config)
            if not hasattr(model, "example_input_array"):
                model.example_input_array = None
            summary = ModelSummary(model)
            n_params = summary.total_parameters
            model_config.n_params = n_params
            if model_path is not None:
                OmegaConf.save(config=model_config, f=model_path)
        if not is_dummy and model_config.type == "gnn":
            mode = model_config.hyperparams.get("mode", "embedding")
            maybe_origin = "Transfer learning" if mode == "embedding" else "Masked modeling"
            maybe_origin = f" ({maybe_origin})"
        else:
            maybe_origin = ""
        n_params = get_human_readable_count(n_params)
        pretty_model_name = (
            f"{n_params} {PRETTY_MODEL_TYPES[model_config.type]}{maybe_origin}{maybe_not_trained}"
            f"{maybe_date}"
        )
    elif model_config.type == "fingerprints":
        dimension = model_config.hyperparams.get("fpSize", 768)
        radius = model_config.hyperparams.get("radius", 2)
        pretty_model_name = f"{dimension}-dim Morgan fingerprints (R={radius})"
    elif model_config.type == "simple_conv":
        orthogonal_init = model_config.hyperparams.get("orthogonal_init", False)
        one_hot = model_config.hyperparams.get("one_hot", False)
        pretty_model_name = (
            "SimpleConv{}".format(
                model_config.hyperparams.get("layers", ""),
            )
            + (" (orthogonal)" if orthogonal_init else "")
            + (" (one-hot)" if one_hot else "")
        )
    elif model_config.type == "mean_embeds":
        one_hot = model_config.hyperparams.get("one_hot", False)
        pretty_model_name = "MeanOneHot" if one_hot else "MeanEmbeds"
    elif model_config.type == "clamp":
        pretty_model_name = "CLAMP"
    elif model_config.type == "descriptors":
        pretty_model_name = "RDKit Descriptors"
    elif model_config.type == "molformer":
        pretty_model_name = "MoLFormer"
    else:
        logger.error("Unknown model type for model %s: %s", model_name, model_config.type)
        return None
    return pretty_model_name
