"""Here we wrap ChemBERTa for use with PyTorch Geometric."""

import logging

import tokenizers
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizerBase

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule, UnknownModelModeError
from utils.load_tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


def check_if_batched(data: Data) -> bool:
    """Check if a Data object contains a batch of SMILES strings.

    Parameters
    ----------
    data : Data
        The Data object to check.

    Returns
    -------
    bool
        True if the Data object contains a batch of SMILES strings, False otherwise.

    """
    if data.smiles is None:
        return False
    return not isinstance(data.smiles, str)


UNKNOWN_LOSS_MSG = "Unknown loss: {}"

@register_model_type("chemberta")
class Chemberta(MultiModalModule, modes=("embedding", "mlm")):
    # TODO: properly wrap with `bind`
    """LightningModule wrapping ChemBERTa for use with PyTorch Geometric."""

    def __init__(  # noqa: PLR0913 # TODO: add config to reduce argument count
        self,
        mode: str,
        tokenizer: AutoTokenizer | tokenizers.Tokenizer,
        roberta_model: RobertaForMaskedLM,
        *,
        normalize: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.9,
        loss: str = "cross_entropy",
        debug: bool = False,
    ) -> None:
        """Initializes a Chemberta model.

        Parameters
        ----------
        mode : str
            The mode of the model. Can be either "embedding" or "classification".
        tokenizer : AutoTokenizer | tokenizers.Tokenizer
            The tokenizer to use for tokenizing the input.
        roberta_model : RobertaForMaskedLM
            The pre-trained ChemBERTa model to use.
        normalize : bool, default=False
            Whether to normalize the output embeddings.
        lr : float, default=3e-4
            The learning rate to use.
        gamma : float, default=0.9
            The gamma value to use for the learning rate scheduler.
        loss: str, default="nll"
            The loss function to use.
        debug : bool, default=False
            Whether to enable debug mode.

        """
        super().__init__(mode)
        self.tokenizer = tokenizer
        self.model = roberta_model
        self.normalize = normalize
        self.lr = lr
        self._gamma = gamma
        if loss.lower() in {"l1", "mae", "l1_loss", "mean_absolute_error", "l1loss"}:
            self.loss = nn.L1Loss()
        elif loss.lower() in {"l2", "mse", "l2_loss", "mean_squared_error", "l2loss"}:
            self.loss = nn.MSELoss()
        elif loss.lower() in {"nll", "negative_log_likelihood"}:
            self.loss = nn.NLLLoss()
        elif loss.lower() in {"ce", "cross_entropy"}:
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(UNKNOWN_LOSS_MSG.format(loss))
        self.debug = debug

    @classmethod
    def from_pretrained(
        cls,
        *,
        normalize: bool = False,
        tokenizer_path_hf: str = "seyonec/ChemBERTa_zinc250k_v2_40k",
        pretrained_model_name_or_path: str = "seyonec/ChemBERTa_zinc250k_v2_40k",
        mode: str = "embedding",
    ) -> "Chemberta":
        """Creates an instance of Chemberta from a pre-trained model.

        Parameters
        ----------
        normalize : bool, default=False
            Whether to normalize the output embeddings.
        tokenizer_path_hf : str, default="seyonec/ChemBERTa_zinc250k_v2_40k"
            Path to the tokenizer on Huggingface.
        pretrained_model_name_or_path : str, default="seyonec/ChemBERTa_zinc250k_v2_40k"
            Path to the model on Huggingface or local path.
        mode : str, default="embedding"
            The mode of the model. Currently only "embedding" is supported.

        Returns
        -------
        Chemberta
            An instance of Chemberta.

        """
        return cls(
            mode=mode,
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_path_hf),
            roberta_model=AutoModel.from_pretrained(pretrained_model_name_or_path),
            normalize=normalize,
        )

    def forward(self, data: Data | dict) -> torch.Tensor:
        """Forward method for Chemberta.

        Parameters
        ----------
        data : Data or dict
            Input data. If Data, it should contain a "smiles" field.
            If dict, it should contain a "smiles" key mapping to a str or a list of str.

        Returns
        -------
        Embeddings or masked language model output, depending on the mode.

        Raises
        ------
        UnknownModelModeError
            If the mode is not "embedding" or "mlm".
        IndexError
            If the input data is empty.
        RuntimeError
            If the input data is invalid.
        AttributeError
            If the input data is missing a "smiles" field.
        KeyError
            If the input data is missing a required key.

        """
        try:
            if self.mode == "embedding":
                return self._forward_embedding_mode(data)
            if self.mode == "mlm":
                return self._forward_mlm_mode(data)
            raise UnknownModelModeError
        except (IndexError, RuntimeError):
            smiles = data.get("smiles", False)
            if smiles:
                if isinstance(smiles, str):
                    logger.exception("Error with smiles %s", smiles)
                else:
                    for this_smiles in smiles:
                        logger.exception("Error with smiles %s", this_smiles)
            raise
        except AttributeError as ae:
            logger.exception("Please, make sure that %s was provided", ae.name)
            raise
        except KeyError as ke:
            logger.exception("Invalid input data format, key %s is missing", ke.args[0])

    def _forward_embedding_mode(self, data: Data | dict) -> None:
        if "smiles" in data:
            smiles = data["smiles"]
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                tokenized = self.tokenizer(smiles, return_tensors="pt", padding=True).to(
                    device=self.model.device,
                )
            elif isinstance(self.tokenizer, tokenizers.Tokenizer):
                if check_if_batched(data):
                    tokenized_tokenizers_format = self.tokenizer.encode_batch(smiles)
                    ids = torch.stack(
                        [
                            torch.tensor(elem.ids, device=self.model.device)
                            for elem in tokenized_tokenizers_format
                        ],
                    )
                    attention_mask = ids != self.tokenizer.padding["pad_id"]
                    tokenized = {
                        "input_ids": ids,
                        "attention_mask": attention_mask,
                    }
                else:
                    tokenized_tokenizers_format = self.tokenizer.encode(smiles)
                    tokenized = {
                        "input_ids": torch.tensor(
                            tokenized_tokenizers_format.ids,
                            device=self.model.device,
                        ),
                    }
        else:
            tokenized = {"input_ids": data["ids"]}
            if "attention_mask" in data:
                tokenized["attention_mask"] = data["attention_mask"]
        if isinstance(self.model, RobertaForMaskedLM):
            out = self.model.roberta(**tokenized)
        else:
            out = self.model(**tokenized)
        embeddings = []
        for embedding_padded, attention_micro_mask in zip(
            out.last_hidden_state,
            tokenized["attention_mask"],
            strict=True,
        ):
            embbedding = embedding_padded[attention_micro_mask].mean(dim=0)
            embeddings.append(embbedding)
        res = torch.stack(embeddings, dim=0)
        if self.normalize:
            res = nn.functional.normalize(res, dim=1)
        return res

    def _forward_mlm_mode(self, data: dict) -> torch.Tensor:
        return self.model(input_ids=data["masked_ids"], attention_mask=data["attention_mask"])

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler._LRScheduler]:
        """Configures the optimizer and learning rate scheduler.

        Returns a dictionary with a single entry, where the key is "optimizer"
        and the value is a torch.optim.Optimizer, and a single entry, where the
        key is "lr_scheduler" and the value is a torch.optim.lr_scheduler.

        The optimizer is a AdamW optimizer with a learning rate of `self.lr`.
        The learning rate scheduler is an ExponentialLR scheduler with a gamma
        of `self._gamma`.

        """
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer,
                gamma=self._gamma,
            ),
        }

    def get_loss(
        self,
        out: torch.Tensor,
        ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the loss given the output of the model and the ids of the batch.

        Parameters
        ----------
        out : torch.Tensor
            output of the model
        ids : torch.Tensor
            ids of the batch
        attention_mask : torch.Tensor
            attention mask

        Returns
        -------
        loss : torch.Tensor
            computed loss

        """
        if isinstance(self.loss, nn.CrossEntropyLoss):
            return self.loss(out[attention_mask], ids[attention_mask])
        ids_onehot = torch.nn.functional.one_hot(ids, num_classes=out.shape[2])
        return self.loss(out[attention_mask], ids_onehot.float()[attention_mask])

    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        """Performs a training step.

        Parameters
        ----------
        batch : dict
            Batch of data to use for training.
        _ : int
            Unused argument.

        Returns
        -------
        loss : torch.Tensor
            Computed loss.

        """
        out = self(batch)
        loss = self.get_loss(out.logits, batch["ids"], batch["attention_mask"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, _: int) -> torch.Tensor:
        """Performs a validation step.

        Parameters
        ----------
        batch : dict
            Batch of data to use for validation.
        _ : int
            Unused argument.

        Returns
        -------
        loss : torch.Tensor
            Computed loss.

        """
        out = self(batch)
        loss = self.get_loss(out.logits, batch["ids"], batch["attention_mask"])
        self.log("val_loss", loss)
        return loss

    @classmethod
    def load_from_config(cls, model_config: OmegaConf) -> "Chemberta":
        """Loads a Chemberta model from its configuration.

        Parameters
        ----------
        model_config : OmegaConf
            Model configuration.

        Returns
        -------
        Chemberta
            Loaded model.

        Notes
        -----
        If the model is "pretrained", it is loaded from its save file.
        Otherwise, a new model is created with the specified configuration.

        """
        hparams = dict(model_config.hyperparams)
        pretrained = model_config.get("pretrained", True)
        ckpt_path = model_config.get("path", None)
        if pretrained and ckpt_path is None:
            return cls.from_pretrained(**hparams)
        hparams["tokenizer"] = load_tokenizer(model_config.tokenizer_name)
        hparams["roberta_model"] = RobertaForMaskedLM(
            config=RobertaConfig(**model_config.roberta_config),
        )
        return cls.load_from_checkpoint(ckpt_path, **hparams) if ckpt_path else cls(**hparams)
