"""Code for MoLFormer-XL model from https://github.com/IBM/molformer."""

import random
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from torch import nn

from nn.molformer.pubchem_encoder import Encoder
from nn.molformer.rotate_attention import RotateEncoderBuilder as rotate_builder  # noqa: N813


class LightningModule(pl.LightningModule):
    def __init__(self, config, vocab, vocab_encoder_weights_path) -> None:
        super(LightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        # location of cache File
        # Special symbols

        self.debug = config.debug
        self.text_encoder = Encoder(
            vocab_encoder_weights_path=vocab_encoder_weights_path,
            max_length=config.max_len,
        )
        # Word embeddings layer
        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd // config.n_head,
            value_dimensions=config.n_embd // config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type="linear",
            # attention_type='full',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation="gelu",
        )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab) -> None:
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)

        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def on_save_checkpoint(self, checkpoint) -> None:
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict["torch_state"] = torch.get_rng_state()
        out_dict["cuda_state"] = torch.cuda.get_rng_state()
        if np:
            out_dict["numpy_state"] = np.random.get_state()
        if random:
            out_dict["python_state"] = random.getstate()
        checkpoint["rng"] = out_dict

    def on_load_checkpoint(self, checkpoint) -> None:
        # load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint["rng"]
        for key, value in rng.items():
            if key == "torch_state":
                torch.set_rng_state(value)
            elif key == "cuda_state":
                torch.cuda.set_rng_state(value)
            elif key == "numpy_state":
                np.random.set_state(value)
            elif key == "python_state":
                random.setstate(value)
            else:
                print("unrecognized state")

    def _init_weights(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def training_step(self, batch, batch_idx):
        idxl = batch[0]
        targetsl = batch[1]
        # lengthsl = batch[2]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            b_element_size = len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            # masking of the length of the inputs its handled in the Masked language part of the code
            # do not attempt to handle it in the forward of the transformer
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs / len(idxl)
            if chunk < len(idxl) - 1:
                loss_tmp.backward()
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log("train_loss", loss, on_step=True)
        return {"loss": loss}  # , 'log':tensorboard_log}

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.tensor([output["loss"] for output in outputs]).mean()
        loss = {"loss": avg_loss.item()}
        self.log("validation_loss", loss["loss"])

    def validation_step(self, batch, batch_idx):
        idxl = batch[0]
        targetsl = batch[1]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            b_element_size = len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs / len(idxl)
            if chunk < len(idxl) - 1:
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log("train_loss", loss, on_step=True)
        return {"loss": loss}
