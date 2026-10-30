"""Wraps a generic sentence-transformers text-embedding model, applied to SMILES as plain text.

Deliberately absurd baseline: a model that has never seen a molecule, reading SMILES strings as
if they were sentences. By default a frozen feature extractor, same shape as
``CheMeleon``/``FingerprintCreator``.

TWO FORWARD PATHS. The frozen path (the default, unchanged) goes through
``SentenceTransformer.encode()``, which runs under ``torch.inference_mode()`` and returns CPU
numpy -- NO GRADIENT CAN REACH THE BACKBONE through it, so a fine-tune built on it silently
trains only the head. ``encode_differentiable()`` is the same computation (prompt -> tokenize ->
backbone -> the model's own pooling/normalisation) with autograd left on, and
``prepare_for_finetuning()`` switches ``forward`` onto it. Frozen embeddings are never produced
by the new path unless that switch is thrown, so every cached ``.npy`` stays what it was.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torch_geometric
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch import nn

from nn._register_model import register_model_type


@register_model_type("text_embedding")
class TextEmbedder(nn.Module):
    """Frozen sentence-transformers text embedder applied to raw SMILES strings."""

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-embeddings-v5-text-small",
        *,
        trust_remote_code: bool = True,
        task: str | None = "text-matching",
    ) -> None:
        """Initializes the TextEmbedder.

        Parameters
        ----------
        model_name_or_path : str, default="jinaai/jina-embeddings-v5-text-small"
            HuggingFace model id or local path for the sentence-transformers model.
        trust_remote_code : bool, default=True
            Passed through to SentenceTransformer; several current text-embedding models
            (e.g. the jina-embeddings-v5 family) ship custom modeling code and require this.
        task : str | None, default="text-matching"
            Task adapter name passed to `.encode()`. jina-embeddings-v5 REQUIRES one
            (verified: `task=None` raises "Task must be specified before encoding data" —
            it is not optional for this model despite being documented as a refinement
            elsewhere). "text-matching" is the closest task-agnostic option and is what the
            model's own usage example uses. None is still accepted for models that make it
            genuinely optional.

        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task = task
        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code)
        self.mode = "embedding"
        # False = frozen encode() path (bit-identical to every cached embedding). Set True only by
        # prepare_for_finetuning(); see the module docstring.
        self.differentiable = False
        # jina-v5 only: True once its task adapter is merged into the weights (the merged backbone
        # is a plain Qwen3Model, which still HAS a set_adapter method -- transformers' PEFT mixin --
        # that raises "No adapter loaded", so the state is tracked here, not probed).
        self._task_adapter_merged = False
        # Read the width off a real encode() call rather than a getter method name, which
        # is mid-rename across sentence-transformers versions (get_sentence_embedding_dimension
        # -> get_embedding_dimension) and may not exist under either name for every backend.
        probe_kwargs = {"convert_to_numpy": True}
        if self.task is not None:
            probe_kwargs["task"] = self.task
        self.output_dim = self.model.encode(["C"], **probe_kwargs).shape[-1]

    def _is_jina_custom_st(self) -> bool:
        """True if the input module is jina-v5's ``custom_st.Transformer``.

        That module's own ``forward`` hard-codes ``self.model.eval()`` and ``torch.no_grad()``, so
        routing through ``SentenceTransformer.forward`` would still cut the graph for jina. Its
        body is replicated (minus those two lines) in ``_jina_forward``.
        """
        first = self.model[0]
        cls = type(first)
        return cls.__name__ == "Transformer" and cls.__module__.endswith("custom_st")

    def _jina_forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """jina-v5 custom_st.Transformer.forward with autograd and train/eval left to the caller.

        Same computation line for line: select the task adapter, run the backbone, take the hidden
        state of the LAST non-pad token, L2-normalise.
        """
        inner = self.model[0].model
        # Before prepare_for_finetuning() the backbone is jina's PeftMixedModel carrying four task
        # adapters, and the task one is selected per call exactly as custom_st does. After it, the
        # task adapter has been merged into the weights and there is nothing left to select.
        if not self._task_adapter_merged:
            task = self.task if self.task is not None else self.model[0].default_task
            inner.set_adapter(task)
        batch = {k: v for k, v in features.items() if torch.is_tensor(v)}
        hidden = inner(**batch).last_hidden_state
        mask = batch.get("attention_mask")
        if mask is None:
            pooled = hidden[:, -1]
        else:
            sequence_lengths = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), sequence_lengths]
        return F.normalize(pooled, p=2, dim=-1)

    def encode_differentiable(self, smiles: list[str]) -> torch.Tensor:
        """Differentiable twin of ``self.model.encode(smiles, task=self.task)``.

        Mirrors encode()'s own steps for one batch -- resolve the DEFAULT PROMPT (jina-v5 prepends
        "Document: " through its ``default_prompt_name``; dropping it would change every
        embedding), ``preprocess``, move to device, forward -- but leaves autograd and train/eval
        mode alone. encode() additionally length-sorts and batches its input; with one batch the
        row order does not change the result (asserted to 1e-5 in
        tests/test_text_embedder_grad.py).

        Returns float32 (jina's backbone is bf16) on the backbone's device.
        """
        kwargs = {} if self.task is None else {"task": self.task}
        prompt = self.model._resolve_prompt(None, None)  # noqa: SLF001 -- the exact rule encode() uses
        features = self.model.preprocess(list(smiles), prompt=prompt, **kwargs)
        features = batch_to_device(features, self.model.device)
        if self._is_jina_custom_st():
            emb = self._jina_forward(features)
        else:
            emb = self.model(features, **kwargs)["sentence_embedding"]
        return emb.float()

    def prepare_for_finetuning(self) -> None:
        """Make the backbone trainable and switch ``forward`` onto ``encode_differentiable``.

        1. UPCAST TO FLOAT32. jina-v5 loads in bfloat16. bf16 keeps ~8 mantissa bits, so an AdamW
           step of lr 5e-6 on a weight of order 1e-2 is below half an ulp and ROUNDS TO NO CHANGE:
           a bf16 full fine-tune at the uniform protocol's lr would silently barely move the
           backbone. The float32 starting function differs from the bf16 frozen embeddings only by
           bf16 rounding. No-op for float32 models (CheMatE).
        2. jina-v5 ONLY -- MERGE THE TASK ADAPTER. The frozen jina features are produced with its
           own LoRA adapter for ``self.task`` ACTIVE (custom_st calls ``set_adapter(task)`` on every
           forward; three other task adapters sit inactive). That adapter is merged into the
           float32 weights and the other three dropped, so the fine-tune starts from exactly the
           function the frozen row measured. Consequence for the PEFT-LoRA arm: OUR adapters are
           injected into the MERGED projections, i.e. they STACK ON TOP of jina's text-matching
           adapter (W + B_tm A_tm + B_ours A_ours, the first two frozen). Without the merge,
           PEFT's ``all-linear`` would reach into jina's own ``lora_A``/``lora_B`` Linears and
           ``set_adapter`` calls would flip which adapter is live mid-run. Under full fine-tuning
           the merged matrix is simply trained.
        """
        self.model.float()
        if self._is_jina_custom_st():
            inner = self.model[0].model
            task = self.task if self.task is not None else self.model[0].default_task
            inner.set_adapter(task)
            self.model[0].model = inner.merge_and_unload(adapter_names=[task])
            self._task_adapter_merged = True
        self.differentiable = True

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Encodes a batch of SMILES strings as text embeddings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing a "smiles" field (list of SMILES strings).

        Returns
        -------
        torch.Tensor
            Text embeddings, shape (batch, self.output_dim).

        """
        if self.differentiable:
            return self.encode_differentiable(list(data.smiles))
        encode_kwargs = {"convert_to_numpy": True}
        if self.task is not None:
            encode_kwargs["task"] = self.task
        embeddings: np.ndarray = self.model.encode(list(data.smiles), **encode_kwargs)
        return torch.tensor(embeddings, dtype=torch.float32,
                            # A record built from SMILES alone (FS-Mol's embed step) has no node
                            # features, so data.x is None and cannot name a device. The output is
                            # moved to CPU by embed_dataset regardless; the main grid always has x
                            # and takes the first branch, exactly as before.
                            device=data.x.device if getattr(data, "x", None) is not None else "cpu")
