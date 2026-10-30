"""ChemFM wrapper: LLaMA-architecture decoder on SMILES (Cai et al., Nature Chemistry 2025).

Single mode (embedding); no bind decorator pattern. Hyperparams not saved via
save_hyperparameters — consistent with codebase.
"""

import logging

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from transformers import AutoModelForCausalLM, AutoTokenizer

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule

logger = logging.getLogger(__name__)


@register_model_type("chemfm")
class LightningChemFM(MultiModalModule, modes=("embedding",)):
    """ChemFM wrapper. Tokenizes SMILES, runs LLaMA decoder, mean-pools last hidden state.

    LLaMA tokenizers often have no native pad token; we assign eos_token as pad_token.
    Mean-pooling over attention_mask excludes pad positions from the embedding.
    """

    def __init__(
        self,
        mode: str = "embedding",
        *,
        pretrained_model_name_or_path: str,
        torch_dtype: str = "float16",
        normalize: bool = False,
    ) -> None:
        super().__init__(mode)
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map[torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            output_hidden_states=True,
        )
        self.normalize = normalize

    def forward(self, data: Data) -> torch.Tensor:
        """Forward: SMILES → tokens → LLaMA → mean-pooled last hidden state."""
        smiles_list = data.smiles if isinstance(data.smiles, list) else [data.smiles]
        tokenized = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        out = self.model(**tokenized, output_hidden_states=True)
        last = out.hidden_states[-1]
        mask = tokenized["attention_mask"].unsqueeze(-1).to(last.dtype)
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeds = (summed / counts).float()
        if self.normalize:
            embeds = nn.functional.normalize(embeds, dim=1)
        return embeds

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "LightningChemFM":
        hparams = dict(model_config.hyperparams)
        ckpt_path = model_config.get("path", None)
        if ckpt_path:
            return cls.load_from_checkpoint(ckpt_path, **hparams)
        return cls(**hparams)
