import logging

import selfies as sf
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule

logger = logging.getLogger(__name__)


@register_model_type("selformer")
class LightningSELFormer(MultiModalModule, modes=("embedding",)):
    """SELFormer wrapper. Converts SMILES → SELFIES, tokenizes, runs RoBERTa, mean-pools."""

    def __init__(
        self,
        mode: str = "embedding",
        *,
        pretrained_model_name_or_path: str,
        tokenizer_name: str | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(mode)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or pretrained_model_name_or_path,
        )
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.normalize = normalize

    def forward(self, data: Data) -> torch.Tensor:
        smiles_list = data.smiles if isinstance(data.smiles, list) else [data.smiles]
        selfies_list: list[str] = []
        for smiles in smiles_list:
            try:
                selfies_list.append(sf.encoder(smiles))
            except sf.EncoderError:
                logger.warning("Could not convert SMILES to SELFIES: %s", smiles)
                selfies_list.append("[C]")
        tokenized = self.tokenizer(
            selfies_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        out = self.model(**tokenized)
        mask = tokenized["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeds = summed / counts
        if self.normalize:
            embeds = nn.functional.normalize(embeds, dim=1)
        return embeds

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "LightningSELFormer":
        hparams = dict(model_config.hyperparams)
        ckpt_path = model_config.get("path", None)
        if ckpt_path:
            return cls.load_from_checkpoint(ckpt_path, **hparams)
        return cls(**hparams)
