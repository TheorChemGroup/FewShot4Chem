import logging

import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule

logger = logging.getLogger(__name__)


@register_model_type("molformer-hf")
class LightningMoLFormerHF(MultiModalModule, modes=("embedding",)):
    """MoLFormer-XL (HuggingFace) wrapper. Mean-pools last hidden state over non-pad tokens."""

    def __init__(
        self,
        mode: str = "embedding",
        *,
        pretrained_model_name_or_path: str = "ibm/MoLFormer-XL-both-10pct",
        trust_remote_code: bool = True,
        deterministic_eval: bool = True,
        normalize: bool = False,
        revision: str | None = None,
        code_revision: str | None = None,
    ) -> None:
        super().__init__(mode)
        # Pin weights and executable code for reproducible, compatible loading.
        # A model can delegate auto_map classes to another repository; revision
        # alone does not pin that repository, so code_revision is also needed.
        # The compat-v4 revision retains tokenizer classes and supports the
        # pinned transformers version; newer code requires a newer version.
        # records, which exist, cannot have been produced under 361063d0.
        self.revision = revision
        self.code_revision = code_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
        )
        # deterministic_eval is IBM-MoLFormer-specific; required for reproducible inference.
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            deterministic_eval=deterministic_eval,
            revision=revision,
            code_revision=code_revision,
        )
        self.normalize = normalize

    def forward(self, data: Data) -> torch.Tensor:
        smiles_list = data.smiles if isinstance(data.smiles, list) else [data.smiles]
        tokenized = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        out = self.model(**tokenized)
        last = out.last_hidden_state
        mask = tokenized["attention_mask"].unsqueeze(-1).to(last.dtype)
        summed = (last * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeds = summed / counts
        if self.normalize:
            embeds = nn.functional.normalize(embeds, dim=1)
        return embeds

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "LightningMoLFormerHF":
        hparams = dict(model_config.hyperparams)
        ckpt_path = model_config.get("path", None)
        if ckpt_path:
            return cls.load_from_checkpoint(ckpt_path, **hparams)
        return cls(**hparams)
