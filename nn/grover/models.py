"""GROVER models for embedding and fingerprint generation.

Adapted from https://github.com/tencent-ailab/grover
"""

from typing import Literal

import numpy as np
import torch
from torch import nn as nn

from nn.grover.layers import GTransEncoder, Readout
from nn.grover.molgraph import GroverArgs, get_atom_fdim, get_bond_fdim


class GROVEREmbedding(nn.Module):
    """GROVER embedding module containing the GTransEncoder."""

    def __init__(self, args: GroverArgs) -> None:
        """Initialize GROVER embedding.

        Parameters
        ----------
        args : GroverArgs
            Model arguments.
        """
        super().__init__()
        self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()
        if not hasattr(args, "backbone"):
            args.backbone = "gtrans"
        if args.backbone == "gtrans" or args.backbone == "dualtrans":
            self.encoders = GTransEncoder(
                args,
                hidden_size=args.hidden_size,
                edge_fdim=edge_dim,
                node_fdim=node_dim,
                dropout=args.dropout,
                activation=args.activation,
                num_mt_block=args.num_mt_block,
                num_attn_head=args.num_attn_head,
                atom_emb_output=self.embedding_output_type,
                bias=args.bias,
                cuda=args.cuda,
            )

    def forward(self, graph_batch: tuple) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        graph_batch : tuple
            Input graph batch from BatchMolGraph.get_components().

        Returns
        -------
        dict[str, torch.Tensor | None]
            Dict containing embeddings based on embedding_output_type.
        """
        output = self.encoders(graph_batch)
        if self.embedding_output_type == "atom":
            return {
                "atom_from_atom": output[0],
                "atom_from_bond": output[1],
                "bond_from_atom": None,
                "bond_from_bond": None,
            }
        elif self.embedding_output_type == "bond":
            return {
                "atom_from_atom": None,
                "atom_from_bond": None,
                "bond_from_atom": output[0],
                "bond_from_bond": output[1],
            }
        elif self.embedding_output_type == "both":
            return {
                "atom_from_atom": output[0][0],
                "bond_from_atom": output[0][1],
                "atom_from_bond": output[1][0],
                "bond_from_bond": output[1][1],
            }
        else:
            return {
                "atom_from_atom": output[0],
                "atom_from_bond": output[1],
                "bond_from_atom": None,
                "bond_from_bond": None,
            }


class GroverFpGeneration(nn.Module):
    """GROVER fingerprint generation module.

    Loads a pre-trained model and produces fingerprints for input molecules.
    """

    def __init__(
        self,
        args: GroverArgs,
        hidden_size: int | None = None,
        fingerprint_source: Literal["atom", "bond", "both"] = "both",
    ) -> None:
        """Initialize fingerprint generator.

        Parameters
        ----------
        args : GroverArgs
            Model arguments.
        hidden_size : int | None, optional
            Hidden size (defaults to args.hidden_size).
        fingerprint_source : str, default="both"
            Source for fingerprint: "atom", "bond", or "both".
        """
        super().__init__()

        self.fingerprint_source = fingerprint_source
        self.iscuda = args.cuda
        self.hidden_size = hidden_size if hidden_size is not None else args.hidden_size

        self.grover = GROVEREmbedding(args)
        self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

    def forward(
        self,
        batch: tuple,
        features_batch: list | None = None,
    ) -> torch.Tensor:
        """Forward pass to generate fingerprints.

        Parameters
        ----------
        batch : tuple
            Graph batch from BatchMolGraph.get_components().
        features_batch : list | None, optional
            Additional molecular features.

        Returns
        -------
        torch.Tensor
            Molecular fingerprints.
        """
        _, _, _, _, _, a_scope, b_scope, _ = batch

        output = self.grover(batch)

        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if self.fingerprint_source == "bond" or self.fingerprint_source == "both":
            mol_bond_from_atom_output = self.readout(output["bond_from_atom"], b_scope)
            mol_bond_from_bond_output = self.readout(output["bond_from_bond"], b_scope)

        if features_batch is not None and features_batch[0] is not None:
            features_batch_tensor = torch.from_numpy(np.stack(features_batch)).float()
            if self.iscuda:
                features_batch_tensor = features_batch_tensor.cuda()
            features_batch_tensor = features_batch_tensor.to(output["atom_from_atom"])
            if len(features_batch_tensor.shape) == 1:
                features_batch_tensor = features_batch_tensor.view([1, features_batch_tensor.shape[0]])
        else:
            features_batch_tensor = None

        if self.fingerprint_source == "atom":
            fp = torch.cat([mol_atom_from_atom_output, mol_atom_from_bond_output], dim=1)
        elif self.fingerprint_source == "bond":
            fp = torch.cat([mol_bond_from_atom_output, mol_bond_from_bond_output], dim=1)
        else:
            fp = torch.cat(
                [
                    mol_atom_from_atom_output,
                    mol_atom_from_bond_output,
                    mol_bond_from_atom_output,
                    mol_bond_from_bond_output,
                ],
                dim=1,
            )
        if features_batch_tensor is not None:
            fp = torch.cat([fp, features_batch_tensor], dim=1)
        return fp

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension.

        Returns
        -------
        int
            Embedding dimension based on fingerprint_source.
        """
        if self.fingerprint_source == "atom":
            return self.hidden_size * 2
        elif self.fingerprint_source == "bond":
            return self.hidden_size * 2
        else:
            return self.hidden_size * 4


class GroverFinetuneTask(nn.Module):
    """GROVER fine-tuning task module."""

    def __init__(
        self,
        args: GroverArgs,
        output_size: int = 1,
        dataset_type: Literal["classification", "regression"] = "regression",
    ) -> None:
        """Initialize fine-tuning task.

        Parameters
        ----------
        args : GroverArgs
            Model arguments.
        output_size : int, default=1
            Output size (number of prediction targets).
        dataset_type : str, default="regression"
            Dataset type: "classification" or "regression".
        """
        super().__init__()

        self.hidden_size = args.hidden_size
        self.iscuda = args.cuda
        self.output_size = output_size

        self.grover = GROVEREmbedding(args)

        self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = self._create_ffn(args)
        self.mol_atom_from_bond_ffn = self._create_ffn(args)

        self.classification = dataset_type == "classification"
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def _create_ffn(self, args: GroverArgs) -> nn.Sequential:
        """Create feed-forward network.

        Parameters
        ----------
        args : GroverArgs
            Model arguments.

        Returns
        -------
        nn.Sequential
            FFN layers.
        """
        first_linear_dim = self.hidden_size

        dropout = nn.Dropout(args.dropout)
        activation = nn.ReLU()

        ffn = nn.Sequential(
            dropout,
            nn.Linear(first_linear_dim, self.hidden_size),
            activation,
            dropout,
            nn.Linear(self.hidden_size, self.output_size),
        )

        return ffn

    def forward(
        self,
        batch: tuple,
        features_batch: list | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        batch : tuple
            Graph batch from BatchMolGraph.get_components().
        features_batch : list | None, optional
            Additional molecular features.

        Returns
        -------
        torch.Tensor
            Predictions.
        """
        _, _, _, _, _, a_scope, _, _ = batch

        output = self.grover(batch)

        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if features_batch is not None and features_batch[0] is not None:
            features_batch_tensor = torch.from_numpy(np.stack(features_batch)).float()
            if self.iscuda:
                features_batch_tensor = features_batch_tensor.cuda()
            features_batch_tensor = features_batch_tensor.to(output["atom_from_atom"])
            if len(features_batch_tensor.shape) == 1:
                features_batch_tensor = features_batch_tensor.view([1, features_batch_tensor.shape[0]])
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch_tensor], dim=1)
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch_tensor], dim=1)

        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)
            output_tensor = (atom_ffn_output + bond_ffn_output) / 2

        return output_tensor