"""MolE wrapper: DeBERTa-on-graphs (Méndez-Lucio et al., Nature Comms 2024).

NOTE: The imports inside __init__ and forward are TEMPLATES based on expected
upstream API (Zenodo record 13891642). When MolE upstream is staged on HPC,
verify actual API and update imports accordingly.

Single mode (embedding); no bind decorator pattern.
"""

import logging
import sys

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule

logger = logging.getLogger(__name__)


@register_model_type("mole")
class LightningMolE(MultiModalModule, modes=("embedding",)):
    """MolE wrapper. Imports upstream mole package from a staged source path."""

    def __init__(
        self,
        mode: str = "embedding",
        *,
        upstream_source_path: str,
        checkpoint_path: str,
    ) -> None:
        super().__init__(mode)
        if upstream_source_path not in sys.path:
            sys.path.insert(0, upstream_source_path)

        # REPLACE THESE IMPORTS WITH ACTUAL UPSTREAM API (see Task 8 Step 1 in plan)
        from mole.model import MolE as UpstreamMolE  # noqa: PLC0415
        from mole.featurizer import featurize_smiles  # noqa: PLC0415

        self._featurize = featurize_smiles
        self.model = UpstreamMolE.from_pretrained(checkpoint_path)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward: SMILES → MolE features → DeBERTa → CLS embedding."""
        smiles_list = data.smiles if isinstance(data.smiles, list) else [data.smiles]
        features = [self._featurize(s) for s in smiles_list]

        # REPLACE WITH ACTUAL UPSTREAM COLLATOR (see Task 8 Step 1 in plan)
        from mole.collator import collate as _collate  # noqa: PLC0415

        batch = _collate(features)
        batch = {k: v.to(next(self.parameters()).device) for k, v in batch.items()}
        out = self.model(**batch)
        return out.last_hidden_state[:, 0, :]

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "LightningMolE":
        hparams = dict(model_config.hyperparams)
        ckpt_path = model_config.get("path", None)
        if ckpt_path:
            return cls.load_from_checkpoint(ckpt_path, **hparams)
        return cls(**hparams)
