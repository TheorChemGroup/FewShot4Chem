"""LightningModule wrapper for Hu et al. GIN-pretrained checkpoints."""

import logging

import torch
from omegaconf import OmegaConf
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from nn._register_model import register_model_type
from nn.gin_pretrained.loader import mol_to_graph_data_obj_simple
from nn.gin_pretrained.model import GNN
from nn.multimodal_module import MultiModalModule

logger = logging.getLogger(__name__)


@register_model_type("gin-pretrained")
class LightningGINPretrained(MultiModalModule, modes=("embedding",)):
    """GIN-pretrained Lightning wrapper. Mean-pools node embeddings → graph embedding."""

    def __init__(
        self,
        mode: str = "embedding",
        *,
        emb_dim: int = 300,
        num_layer: int = 5,
        drop_ratio: float = 0.0,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__(mode)
        self.gnn = GNN(num_layer=num_layer, emb_dim=emb_dim, drop_ratio=drop_ratio)
        if checkpoint_path:
            # Hu et al. checkpoints predate `weights_only` and contain non-tensor pickle metadata.
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_keys = list(ckpt.keys())
            if ckpt_keys and ckpt_keys[0].startswith("gnn."):
                state_dict = {k.removeprefix("gnn."): v for k, v in ckpt.items()}
            else:
                state_dict = ckpt
            missing, unexpected = self.gnn.load_state_dict(state_dict, strict=False)
            if unexpected:
                msg = f"GIN load: unexpected keys (architecture mismatch): {unexpected}"
                raise RuntimeError(msg)
            critical_missing = [k for k in missing if "embedding" in k or "gnns" in k or "batch_norms" in k]
            if critical_missing:
                msg = f"GIN load: critical keys missing from checkpoint: {critical_missing}"
                raise RuntimeError(msg)
            logger.info("Loaded GIN weights from %s (missing non-critical: %s)", checkpoint_path, missing)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Forward: SMILES → PyG graph → GIN → global mean pool → 300-dim embedding."""
        if hasattr(data, "smiles") and data.smiles is not None:
            smiles_list = data.smiles if isinstance(data.smiles, list) else [data.smiles]
            graphs = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning("Invalid SMILES (RDKit failed to parse): %s", smiles)
                    continue
                graphs.append(mol_to_graph_data_obj_simple(mol))
            if not graphs:
                return torch.zeros((0, self.gnn.emb_dim), device=next(self.gnn.parameters()).device)
            batch = Batch.from_data_list(graphs).to(next(self.gnn.parameters()).device)
        else:
            batch = data
        node_embeds = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        return global_mean_pool(node_embeds, batch.batch)

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "LightningGINPretrained":
        hparams = dict(model_config.hyperparams)
        ckpt_path = model_config.get("path", None)
        if ckpt_path:
            return cls.load_from_checkpoint(ckpt_path, **hparams)
        return cls(**hparams)
