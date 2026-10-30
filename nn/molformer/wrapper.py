"""Provides compatible with the current project wrapper for the Molformer model."""

from argparse import Namespace
from pathlib import Path

import torch
import torch_geometric
from fast_transformers.masking import LengthMask
from omegaconf import OmegaConf
from torch_geometric.data import download_google_url, extract_zip

from nn._register_model import load_from_config_default, register_model_type
from nn.molformer.original import LightningModule as OriginalMolformer
from nn.molformer.tokenizer import MolTranBertTokenizer
from nn.multimodal_module import MultiModalModule


def download_pretrained() -> None:
    """Downloads the pre-trained Molformer model weights and the vocabulary file for the tokenizer.

    The pre-trained weights are downloaded from Google Drive and stored in the
    "models/weights/molformer"
    directory. The vocabulary file is downloaded from Google Drive and stored in the
    "models/configs/tokenizers/vocabs" directory.

    If the pre-trained weights or vocabulary file already exist in the specified directories, this
    function does not download them again.

    """
    molformer_weights_folder = Path("models", "weights", "molformer")
    molformer_archive_name = "molformer.zip"
    molformer_weights_file_internal_path = Path(
        "Pretrained MoLFormer",
        "checkpoints",
        "N-Step-Checkpoint_3_30000.ckpt",
    )
    molformer_weights_file_name = "N-Step-Checkpoint_3_30000.ckpt"
    if not (molformer_weights_folder / molformer_weights_file_name).exists():
        molformer_weights_folder.mkdir(parents=True, exist_ok=True)
        if not (molformer_weights_folder / molformer_weights_file_internal_path).exists():
            if not (molformer_weights_folder / molformer_archive_name).exists():
                download_google_url(
                    "1eSDknh11hWOICsh3E_ZVkBwl4rlyKjJ5",
                    molformer_weights_folder,
                    molformer_archive_name,
                )
            extract_zip(
                molformer_weights_folder / molformer_archive_name,
                molformer_weights_folder,
            )

        (molformer_weights_folder / molformer_weights_file_internal_path).rename(
            molformer_weights_folder / molformer_weights_file_name,
        )

    molformer_encode_weights_path = Path(
        "models",
        "weights",
        "molformer",
        "encoder",
        "pubchem_canon_zinc_final_vocab_sorted.pth",
    )

    molformer_encode_weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not molformer_encode_weights_path.exists():
        download_google_url(
            "1LbP35wZrhrGb5B9hBPLauaufLZVG9ZjA",
            molformer_encode_weights_path.parent,
            molformer_encode_weights_path.name,
        )

    bert_vocab_path = Path("models", "configs", "tokenizers", "vocabs", "bert_vocab.txt")
    if not bert_vocab_path.exists():
        download_google_url(
            "1J5h-kZWoQOg_1Cyzdv4b6Dw5DOiDvWE_",
            bert_vocab_path.parent,
            bert_vocab_path.name,
        )


@register_model_type("molformer")
class LightningMoLFormer(MultiModalModule, modes=("embedding",)):
    """LightningMoLformer model.

    Parameters
    ----------
    config : OmegaConf
        Model configuration.
    bert_vocab_path : str
        Path to the vocabulary file for the MolTranBertTokenizer.
    vocab_encoder_weights_path : str
        Path to the pre-trained weights file for the Molformer model.

    Notes
    -----
    For loading from original checkpoints:

    The pre-trained weights can be downloaded from Google Drive and stored in the
    "models/weights/molformer"
    directory. The vocabulary file is downloaded from Google Drive and stored in the
    "models/configs/tokenizers/vocabs" directory.

    If the pre-trained weights or vocabulary file already exist in the specified directories, this
    function does not download them again.

    """

    def __init__(
        self,
        config: OmegaConf,
        bert_vocab_path: str,
        vocab_encoder_weights_path: str,
    ) -> None:
        """Initializes the LightningMoLFormer model.

        Parameters
        ----------
        config : OmegaConf
            Model configuration.
        bert_vocab_path : str
            Path to the vocabulary file for the MolTranBertTokenizer.
        vocab_encoder_weights_path : str
            Path to the pre-trained weights file for the Molformer model.

        Returns
        -------
        None

        """
        super().__init__(mode="embedding")
        bert_vocab_path = Path(bert_vocab_path)
        self.tokenizer = MolTranBertTokenizer(
            str(
                bert_vocab_path.absolute().resolve(),
            ),
        )

        namespace = Namespace(**config)
        self.model = OriginalMolformer(
            config=namespace,
            vocab=self.tokenizer.vocab,
            vocab_encoder_weights_path=vocab_encoder_weights_path,
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Forward pass of the Molformer model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing a "smiles" field.

        Returns
        -------
        torch.Tensor
            Sum of the token embeddings of the input SMILES string, divided by the sum of the input mask.

        """
        smiles_list = data.smiles
        batch_enc = self.tokenizer.batch_encode_plus(
            smiles_list,
            padding=True,
            add_special_tokens=True,
        )
        idx, mask = torch.tensor(batch_enc["input_ids"]), torch.tensor(batch_enc["attention_mask"])
        token_embeddings = self.model.blocks(
            self.model.tok_emb(idx),
            length_mask=LengthMask(mask.sum(-1)),
        )
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @classmethod
    def load_from_config(cls, config: OmegaConf) -> None:
        """Loads a MoLFormer model from its configuration.

        Parameters
        ----------
        config : OmegaConf
            Model configuration.

        Returns
        -------
        Molformer
            Loaded model.

        Notes
        -----
        If `config.legacy_checkpoint` is True, the model is loaded from a legacy checkpoint.
        Otherwise, the model is loaded using `load_from_config_default`.

        """
        if config.legacy_checkpoint:
            download_pretrained()
            model = cls(**config.hyperparams)
            model.model = OriginalMolformer.load_from_checkpoint(
                checkpoint_path=config.path,
                config=Namespace(**config.hyperparams.config),
                vocab=model.tokenizer.vocab,
                vocab_encoder_weights_path=config.hyperparams.vocab_encoder_weights_path,
            )
            return model

        return load_from_config_default(cls, config)
