"""MLPHead — 2-hidden-layer MLP with PyTorch Lightning, early stopping."""

import logging
import os
from pathlib import Path
from typing import Any

import lightning as L  # noqa: N812
import numpy as np
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.plugins.environments import LightningEnvironment
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from validation.heads._register_head import register_head_type
from validation.heads.abstract import Head

logger = logging.getLogger(__name__)

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class _MLPLightning(L.LightningModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        task: str,
        hidden_dim: int,
        n_hidden_layers: int,
        activation: str,
        dropout: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}. Valid options: {sorted(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(prev, hidden_dim), act_cls(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            # fit() reshapes 1-D targets to (n, 1); squeeze the trailing dim so the
            # target rank matches the squeezed logits (binary) / class-index vector.
            y = y.squeeze(-1) if y.ndim > 1 and y.shape[-1] == 1 else y
            if logits.shape[1] == 1:
                return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
            return nn.functional.cross_entropy(logits, y.long())
        return nn.functional.mse_loss(logits, y.float())

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = self._loss(self(x), y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = self._loss(self(x), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )


@register_head_type("mlp")
class MLPHead(Head):
    """Lightning MLP head. Supports regression (single/multi-target) and binary classification."""

    def __init__(self, head_config: OmegaConf) -> None:
        self.head_config = head_config
        self.task = head_config.objective_type
        self._model: _MLPLightning | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        **_: Any,
    ) -> None:
        if sample_weight is not None:
            logger.warning(
                "MLPHead does not support sample_weight; ignoring. "
                "Classification metrics on imbalanced datasets may be biased.",
            )
        L.seed_everything(int(self.head_config.seed), workers=True)

        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train,
                y_train,
                test_size=float(self.head_config.val_fraction),
                random_state=int(self.head_config.seed),
            )

        y_train_arr = np.asarray(y_train)
        y_val_arr = np.asarray(y_val)
        if y_train_arr.ndim == 1:
            y_train_arr = y_train_arr.reshape(-1, 1)
            y_val_arr = y_val_arr.reshape(-1, 1)

        out_dim = y_train_arr.shape[1] if self.task == "regression" else 1

        self._model = _MLPLightning(
            in_dim=x_train.shape[1],
            out_dim=out_dim,
            task=self.task,
            hidden_dim=int(self.head_config.hidden_dim),
            n_hidden_layers=int(self.head_config.n_hidden_layers),
            activation=str(self.head_config.activation),
            dropout=float(self.head_config.dropout),
            lr=float(self.head_config.lr),
            weight_decay=float(self.head_config.weight_decay),
        )

        dtype = np.float32 if self.task == "regression" else np.int64
        train_ds = TensorDataset(
            torch.from_numpy(np.asarray(x_train, dtype=np.float32)),
            torch.from_numpy(y_train_arr.astype(dtype)),
        )
        val_ds = TensorDataset(
            torch.from_numpy(np.asarray(x_val, dtype=np.float32)),
            torch.from_numpy(y_val_arr.astype(dtype)),
        )
        bs = int(self.head_config.batch_size)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

        trainer = L.Trainer(
            max_epochs=int(self.head_config.max_epochs),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=int(self.head_config.early_stopping_patience),
                    mode="min",
                ),
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            # Selecting CPU avoids Lightning's CUDA/NVML probe, which can block
            # when the GPU driver is unresponsive. The environment variable
            # overrides the head configuration; the default is automatic.
            accelerator=os.environ.get("A10_MLP_ACCEL") or str(self.head_config.get("accelerator", "auto")),
            devices=1,
            # This head is single-process; prevent Lightning from launching
            # distributed workers through an auto-detected scheduler.
            plugins=[LightningEnvironment()],
        )
        # Fail rather than hang if a distributed worker is unexpectedly launched.
        if int(getattr(trainer, "world_size", 1)) != 1:
            cenv = type(getattr(trainer.strategy, "cluster_environment", None)).__name__
            msg = (
                f"MLPHead Trainer is distributed (world_size={trainer.world_size}, env={cenv}) — "
                f"refusing; expected single-process LightningEnvironment"
            )
            raise RuntimeError(msg)
        logger.info(
            "MLP Trainer single-process OK (world_size=1, env=%s)",
            type(getattr(trainer.strategy, "cluster_environment", None)).__name__,
        )
        trainer.fit(self._model, train_loader, val_loader)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self._model is None:
            msg = "MLPHead.predict called before fit"
            raise RuntimeError(msg)
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(x_test, dtype=np.float32))
            logits = self._model(x).cpu().numpy()
        if self.task == "classification":
            if logits.shape[1] == 1:
                return (logits.squeeze(-1) > 0).astype(np.int64)
            return logits.argmax(axis=1)
        if logits.shape[1] == 1:
            return logits.squeeze(-1)
        return logits

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray | None:
        if self.task != "classification":
            return None
        if self._model is None:
            msg = "MLPHead.predict_proba called before fit"
            raise RuntimeError(msg)
        self._model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(x_test, dtype=np.float32))
            logits = self._model(x).cpu().numpy()
        if logits.shape[1] == 1:
            p1 = 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))
            return np.stack([1 - p1, p1], axis=1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        if self._model is None:
            msg = "MLPHead.save called before fit"
            raise RuntimeError(msg)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._model.state_dict(), "hparams": dict(self._model.hparams)}, path)
