"""Lightning wrapper for GROVER model.

This module provides a PyTorch Lightning wrapper for the GROVER model,
enabling it to be used with the existing chembertoid pipeline.
"""

import logging
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch_geometric
from omegaconf import OmegaConf
from torch_geometric.data import download_google_url

from nn._register_model import register_model_type
from nn.grover.models import GroverFpGeneration
from nn.grover.molgraph import GroverArgs
from nn.grover_adapter import PyGToGroverAdapter

logger = logging.getLogger(__name__)

GROVER_URL_GOOGLE_IDS = {
    "base": "1HWrAN81_6rnL5mFxPzfxBu-t8Tx4Dgni",
    "large": "1dIs7Eoeb8HxBc2sE5TvAZto1yY1EuJn0",
}

GROVER_CONFIGS = {
    "base": {
        "hidden_size": 1024,
        "depth": 5,
        "num_mt_block": 1,
        "num_attn_head": 8,
        "dropout": 0.1,
        "activation": "PReLU",
    },
    "large": {
        "hidden_size": 1536,
        "depth": 5,
        "num_mt_block": 1,
        "num_attn_head": 16,
        "dropout": 0.1,
        "activation": "PReLU",
    },
}

PROVIDE_HYPERPARAMS_ERROR_MESSAGE = (
    "If model_size is None, all of hidden_size, depth, num_mt_block, "
    "and num_attn_head must be provided."
)


@register_model_type("grover")
class LightningGrover(L.LightningModule):
    """PyTorch Lightning wrapper for GROVER model.

    This wrapper enables GROVER to work with PyTorch Geometric data pipelines
    by converting PyG Data objects to GROVER's internal format.

    Parameters
    ----------
    model_size : str
        Model size: "base" or "large".
    hidden_size : int
        Hidden size of the model.
    depth : int
        Message passing depth.
    num_mt_block : int
        Number of multi-head attention blocks.
    num_attn_head : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    activation : str
        Activation function name.
    bias : bool
        Whether to use bias in linear layers.
    undirected : bool
        Whether to use undirected message passing.
    dense : bool
        Whether to use dense connections.
    fingerprint_source : str
        Source for fingerprint: "atom", "bond", or "both".
    lr : float
        Learning rate.
    gamma : float
        Gamma for learning rate scheduler.
    pretrained : bool
        Whether to load pretrained weights.
    checkpoint_path : str | None
        Path to checkpoint file.

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_size: Literal["base", "large"] | None = "base",
        hidden_size: int | None = None,
        depth: int | None = None,
        num_mt_block: int | None = None,
        num_attn_head: int | None = None,
        dropout: float = 0.1,
        activation: str = "PReLU",
        bias: bool = False,
        undirected: bool = True,
        dense: bool = True,
        fingerprint_source: Literal["atom", "bond", "both"] = "both",
        lr: float = 3e-4,
        gamma: float = 0.95,
    ) -> None:
        """Initialize LightningGrover.

        Parameters
        ----------
        model_size : str | None, default="base", None for custom model
            Model size: "base" or "large".
        hidden_size : int | None, optional
            Hidden size (defaults based on model_size).
        depth : int | None, optional
            Message passing depth (defaults based on model_size).
        num_mt_block : int | None, optional
            Number of MT blocks.
        num_attn_head : int | None, optional
            Number of attention heads.
        dropout : float, default=0.1
            Dropout rate.
        activation : str, default="PReLU"
            Activation function.
        bias : bool, default=False
            Use bias.
        undirected : bool, default=True
            Undirected message passing.
        dense : bool, default=True
            Dense connections.
        fingerprint_source : str, default="both"
            Fingerprint source.
        lr : float, default=3e-4
            Learning rate.
        gamma : float, default=0.95
            LR scheduler gamma.

        """
        super().__init__()
        config = GROVER_CONFIGS[model_size] if model_size is not None else {}
        try:
            self.hidden_size = hidden_size if hidden_size is not None else config["hidden_size"]
            self.depth = depth if depth is not None else config["depth"]
            self.num_mt_block = num_mt_block if num_mt_block is not None else config["num_mt_block"]
            self.num_attn_head = (
                num_attn_head if num_attn_head is not None else config["num_attn_head"]
            )
        except KeyError as key_error:
            raise ValueError(PROVIDE_HYPERPARAMS_ERROR_MESSAGE) from key_error
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.undirected = undirected
        self.dense = dense
        self.fingerprint_source = fingerprint_source
        self.lr = lr
        self.gamma = gamma
        self.model_size = model_size

        self.args = GroverArgs(
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            depth=self.depth,
            num_mt_block=self.num_mt_block,
            num_attn_head=self.num_attn_head,
            bias=self.bias,
            undirected=self.undirected,
            dense=self.dense,
            cuda=False,
            bond_drop_rate=0.0,
            backbone="gtrans",
            embedding_output_type="both",
            fingerprint_source=self.fingerprint_source,
        )

        self.adapter = PyGToGroverAdapter(
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            depth=self.depth,
            num_mt_block=self.num_mt_block,
            num_attn_head=self.num_attn_head,
            bias=self.bias,
            undirected=self.undirected,
            dense=self.dense,
            cuda=False,
        )

        self.model = GroverFpGeneration(
            args=self.args,
            hidden_size=self.hidden_size,
            fingerprint_source=self.fingerprint_source,
        )

        self._optimizer = None

    def forward(self, data: torch_geometric.data.Data | torch_geometric.data.Batch) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data | torch_geometric.data.Batch
            Input data with 'smiles' attribute.

        Returns
        -------
        torch.Tensor
            Molecular embeddings/fingerprints.

        """
        grover_batch = self.adapter.convert(data)
        return self.model(grover_batch, features_batch=None)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension.

        Returns
        -------
        int
            Embedding dimension.

        """
        return self.model.get_embedding_dim()

    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        """Training step.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss tensor.

        """
        embeddings = self(batch)
        if hasattr(batch, "y") and batch.y is not None:
            loss = torch.nn.functional.mse_loss(embeddings, batch.y)
            self.log("train_loss", loss)
            return loss
        return embeddings.mean() * 0

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int) -> torch.Tensor:
        """Validation step.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss tensor.

        """
        embeddings = self(batch)
        if hasattr(batch, "y") and batch.y is not None:
            loss = torch.nn.functional.mse_loss(embeddings, batch.y)
            self.log("val_loss", loss)
            return loss
        return embeddings.mean() * 0

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizers and schedulers.

        Returns
        -------
        dict
            Optimizer configuration.

        """
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer,
                gamma=self.gamma,
            ),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_size: Literal["base", "large"] = "base",
        *,
        fingerprint_source: Literal["atom", "bond", "both"] = "both",
        cache_dir: str | None = None,
        force_download: bool = False,
    ) -> "LightningGrover":
        """Load a pretrained GROVER model.

        Parameters
        ----------
        model_size : str, default="base"
            Model size: "base" or "large".
        fingerprint_source : str, default="both"
            Fingerprint source.
        cache_dir : str | None, default=None
            Directory to cache downloaded weights.
        force_download : bool, default=False
            Force redownload even if file exists.

        Returns
        -------
        LightningGrover
            Loaded model.

        Raises
        ------
        RuntimeError
            If pretrained weights cannot be downloaded or found.

        """
        config = GROVER_CONFIGS.get(model_size, GROVER_CONFIGS["base"])

        model = cls(
            model_size=model_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_mt_block=config["num_mt_block"],
            num_attn_head=config["num_attn_head"],
            dropout=config["dropout"],
            activation=config["activation"],
            fingerprint_source=fingerprint_source,
        )

        checkpoint_path = cls._get_pretrained_checkpoint(model_size, cache_dir, force_download)
        state_dict = cls._load_grover_checkpoint(checkpoint_path)
        if state_dict is not None:
            model._load_state_dict_with_mapping(state_dict)
            logger.info("Loaded pretrained GROVER %s weights", model_size)

        return model

    @staticmethod
    def _get_pretrained_checkpoint(
        model_size: str,
        cache_dir: str | None = None,
        force_download: bool = False,
    ) -> Path:
        """Get path to pretrained checkpoint, downloading if necessary.

        Parameters
        ----------
        model_size : str
            Model size: "base" or "large".
        cache_dir : str | None, optional
            Cache directory.
        force_download : bool, default=False
            Force redownload even if file exists.

        Returns
        -------
        Path
            Path to checkpoint file.

        Raises
        ------
        RuntimeError
            If checkpoint cannot be downloaded or extracted.

        """
        if cache_dir is None:
            cache_dir = "~/.cache/grover"

        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)

        checkpoint_name = f"model_{model_size}.pt"
        checkpoint_path = cache_path / checkpoint_name

        if not force_download and checkpoint_path.exists():
            return checkpoint_path
        checkpoint_google_id = GROVER_URL_GOOGLE_IDS[model_size]
        download_google_url(checkpoint_google_id, folder=str(cache_path), filename=checkpoint_name)
        return checkpoint_path

    @staticmethod
    def _load_grover_checkpoint(checkpoint_path: Path) -> dict | None:
        """Load state dict from GROVER checkpoint.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file.

        Returns
        -------
        dict | None
            State dict or None if loading failed.

        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    return checkpoint["model_state_dict"]
                if "state_dict" in checkpoint:
                    return checkpoint["state_dict"]
                return checkpoint

        except Exception as e:
            logger.warning("Failed to load GROVER checkpoint: %s", e)
            return None
        else:
            return None

    def _load_state_dict_with_mapping(self, state_dict: dict) -> None:
        """Load state dict with key mapping.

        Parameters
        ----------
        state_dict : dict
            State dict to load.

        """
        model_state_dict = self.model.state_dict()

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Handle different checkpoint formats
            if key.startswith("model.grover."):
                # Remove the 'model.' prefix
                new_key = key[6:]
            elif key.startswith("grover."):
                # Already correct format
                new_key = key
            elif key.startswith("encoders."):
                # Add grover prefix
                new_key = "grover.encoders." + key[9:]

            # Try exact match first
            if new_key in model_state_dict:
                if model_state_dict[new_key].shape == value.shape:
                    new_state_dict[new_key] = value
            # Try original key if new_key didn't match
            elif key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    new_state_dict[key] = value

        if new_state_dict:
            self.model.load_state_dict(new_state_dict, strict=False)

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "LightningGrover":
        """Load model from configuration.

        Parameters
        ----------
        model_config : OmegaConf
            Model configuration.

        Returns
        -------
        LightningGrover
            Loaded model.

        """
        hparams = dict(model_config.get("hyperparams", {}))

        pretrained = model_config.get("pretrained", True)
        checkpoint_path = model_config.get("path", None)

        if pretrained and checkpoint_path is None:
            return cls.from_pretrained(
                **hparams,
            )

        if checkpoint_path:
            return cls.load_from_checkpoint(checkpoint_path, **hparams)

        return cls(**hparams)
