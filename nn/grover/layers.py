"""Basic building blocks for GROVER model.

Adapted from https://github.com/tencent-ailab/grover
"""

import math
from typing import Union

import numpy as np
import scipy.stats as stats
import torch
from torch import nn as nn
from torch.nn import LayerNorm, functional as F

from nn.grover.nn_utils import get_activation_function, select_neighbor_and_aggregate


class SelfAttention(nn.Module):
    """Self-attention layer for graph-level readout.

    Given X of shape (n, in_feature), the attention is computed as:
    a = Softmax(W2 @ tanh(W1 @ X.T)), where W1 is (hidden, in_feature) and
    W2 is (out_feature, hidden).

    The final output is: out = a @ X, which is unrelated with input n.
    """

    def __init__(self, *, hidden: int, in_feature: int, out_feature: int) -> None:
        """Initialize self-attention layer.

        Parameters
        ----------
        hidden : int
            Hidden dimension (number of experts).
        in_feature : int
            Input feature dimension.
        out_feature : int
            Output feature dimension.
        """
        super().__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(hidden, in_feature))
        self.w2 = nn.Parameter(torch.FloatTensor(out_feature, hidden))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use xavier_normal method to initialize parameters."""
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Input feature map of shape (n, in_feature).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output embeddings and attention matrix.
        """
        x = torch.tanh(torch.matmul(self.w1, X.transpose(1, 0)))
        x = torch.matmul(self.w2, x)
        attn = F.softmax(x, dim=-1)
        x = torch.matmul(attn, X)
        return x, attn


class Readout(nn.Module):
    """Readout function to convert node embeddings to graph embeddings."""

    def __init__(
        self,
        rtype: str = "none",
        hidden_size: int = 0,
        attn_hidden: int | None = None,
        attn_out: int | None = None,
    ) -> None:
        """Initialize readout.

        Parameters
        ----------
        rtype : str, default="none"
            Readout type: "mean" or "self_attention".
        hidden_size : int, default=0
            Input hidden size.
        attn_hidden : int | None, default=None
            Attention hidden size (for self_attention).
        attn_out : int | None, default=None
            Attention output size (for self_attention).
        """
        super().__init__()
        self.cached_zero_vector = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.rtype = "mean"

        if rtype == "self_attention":
            self.attn = SelfAttention(
                hidden=attn_hidden,
                in_feature=hidden_size,
                out_feature=attn_out,
            )
            self.rtype = "self_attention"
        self.attn_hidden = attn_hidden
        self.attn_out = attn_out

    def forward(
        self, embeddings: torch.Tensor, scope: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embedding matrix, num_atoms or num_bonds x hidden_size.
        scope : list[tuple[int, int]]
            List of (start, range) for each molecule.

        Returns
        -------
        torch.Tensor
            Graph-level embeddings.
        """
        mol_vecs = []
        for _, (a_start, a_size) in enumerate(scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = embeddings.narrow(0, a_start, a_size)
                if self.rtype == "self_attention":
                    cur_hiddens, _ = self.attn(cur_hiddens)
                    cur_hiddens = cur_hiddens.flatten()
                else:
                    cur_hiddens = cur_hiddens.sum(dim=0) / a_size
                mol_vecs.append(cur_hiddens)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs


class MPNEncoder(nn.Module):
    """Message passing neural network encoder."""

    def __init__(
        self,
        args,
        atom_messages: bool,
        init_message_dim: int,
        attached_fea_fdim: int,
        hidden_size: int,
        bias: bool,
        depth: int,
        dropout: float,
        undirected: bool,
        dense: bool,
        aggregate_to_atom: bool,
        attach_fea: bool,
        input_layer: str = "fc",
        dynamic_depth: str = "none",
    ) -> None:
        """Initialize MPNEncoder.

        Parameters
        ----------
        args : GroverArgs
            Arguments.
        atom_messages : bool
            Use atom messages or bond messages.
        init_message_dim : int
            Initial input message dimension.
        attached_fea_fdim : int
            Attached feature dimension.
        hidden_size : int
            Output message dimension.
        bias : bool
            Use bias in message passing.
        depth : int
            Message passing depth.
        dropout : float
            Dropout rate.
        undirected : bool
            Undirected message passing.
        dense : bool
            Dense connections.
        aggregate_to_atom : bool
            Aggregate to atoms.
        attach_fea : bool
            Attach features during message passing.
        input_layer : str, default="fc"
            Input layer type.
        dynamic_depth : str, default="none"
            Dynamic depth mode.
        """
        super().__init__()
        self.init_message_dim = init_message_dim
        self.attached_fea_fdim = attached_fea_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.input_layer = input_layer
        self.layers_per_message = 1
        self.undirected = undirected
        self.atom_messages = atom_messages
        self.dense = dense
        self.aggreate_to_atom = aggregate_to_atom
        self.attached_fea = attach_fea
        self.dynamic_depth = dynamic_depth

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)

        if self.input_layer == "fc":
            input_dim = self.init_message_dim
            self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.attached_fea:
            w_h_input_size = self.hidden_size + self.attached_fea_fdim
        else:
            w_h_input_size = self.hidden_size

        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

    def forward(
        self,
        init_messages: torch.Tensor,
        init_attached_features: torch.Tensor,
        a2nei: torch.Tensor,
        a2attached: torch.Tensor,
        b2a: torch.Tensor | None = None,
        b2revb: torch.Tensor | None = None,
        adjs: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        Parameters
        ----------
        init_messages : torch.Tensor
            Initial messages (atom or bond features).
        init_attached_features : torch.Tensor
            Initial attached features.
        a2nei : torch.Tensor
            Relation of item to its neighbors.
        a2attached : torch.Tensor
            Relation to attached features.
        b2a : torch.Tensor | None, optional
            Bond to atom mapping.
        b2revb : torch.Tensor | None, optional
            Bond to reverse bond mapping.
        adjs : torch.Tensor | None, optional
            Adjacency information.

        Returns
        -------
        torch.FloatTensor
            Output embeddings.
        """
        if self.input_layer == "fc":
            input_msg = self.W_i(init_messages)
            message = self.act_func(input_msg)
        elif self.input_layer == "none":
            input_msg = init_messages
            message = input_msg

        attached_fea = init_attached_features

        if self.training and self.dynamic_depth != "none":
            if self.dynamic_depth == "uniform":
                ndepth = np.random.randint(self.depth - 3, self.depth + 3)
            else:
                mu = self.depth
                sigma = 1
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma
                X = stats.truncnorm(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
                )
                ndepth = int(X.rvs(1))
        else:
            ndepth = self.depth

        for _ in range(ndepth - 1):
            if self.undirected and not self.atom_messages and b2revb is not None:
                message = (message + message[b2revb]) / 2

            nei_message = select_neighbor_and_aggregate(message, a2nei)
            a_message = nei_message
            if self.attached_fea:
                attached_nei_fea = select_neighbor_and_aggregate(attached_fea, a2attached)
                a_message = torch.cat((nei_message, attached_nei_fea), dim=1)

            if not self.atom_messages and b2revb is not None:
                rev_message = message[b2revb]
                if self.attached_fea and b2a is not None:
                    atom_rev_message = attached_fea[b2a[b2revb]]
                    rev_message = torch.cat((rev_message, atom_rev_message), dim=1)
                if b2a is not None:
                    message = a_message[b2a] - rev_message
                else:
                    message = a_message
            else:
                message = a_message

            message = self.W_h(message)

            if self.dense:
                message = self.act_func(message)
            else:
                message = self.act_func(input_msg + message)
            message = self.dropout_layer(message)

        output = message
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "PReLU",
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        """Initialize FFN.

        Parameters
        ----------
        d_model : int
            Input dimension.
        d_ff : int
            Hidden dimension.
        activation : str, default="PReLU"
            Activation function.
        dropout : float, default=0.1
            Dropout rate.
        d_out : int | None, default=None
            Output dimension (defaults to d_model).
        """
        super().__init__()
        if d_out is None:
            d_out = d_model
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.act_func = get_activation_function(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))


class SublayerConnection(nn.Module):
    """Residual connection followed by layer norm.

    Note: The norm is first as opposed to last for code simplicity.
    """

    def __init__(self, size: int, dropout: float) -> None:
        """Initialize sublayer connection.

        Parameters
        ----------
        size : int
            Input dimension.
        dropout : float
            Dropout ratio.
        """
        super().__init__()
        self.norm = LayerNorm(size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, inputs: torch.Tensor | None, outputs: torch.Tensor
    ) -> torch.Tensor:
        """Apply residual connection.

        Parameters
        ----------
        inputs : torch.Tensor | None
            Input tensor for residual connection.
        outputs : torch.Tensor
            Output from sublayer.

        Returns
        -------
        torch.Tensor
            Output with residual connection.
        """
        if inputs is None:
            return self.dropout(self.norm(outputs))
        return inputs + self.dropout(self.norm(outputs))


class Attention(nn.Module):
    """Scaled dot product attention."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor.
        key : torch.Tensor
            Key tensor.
        value : torch.Tensor
            Value tensor.
        mask : torch.Tensor | None, optional
            Attention mask.
        dropout : nn.Dropout | None, optional
            Dropout layer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output and attention weights.
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1, bias: bool = False) -> None:
        """Initialize multi-head attention.

        Parameters
        ----------
        h : int
            Number of attention heads.
        d_model : int
            Model dimension.
        dropout : float, default=0.1
            Dropout rate.
        bias : bool, default=False
            Use bias in output linear layer.
        """
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor.
        key : torch.Tensor
            Key tensor.
        value : torch.Tensor
            Value tensor.
        mask : torch.Tensor | None, optional
            Attention mask.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Head(nn.Module):
    """One head for multi-headed attention using MPN encoders."""

    def __init__(self, args, hidden_size: int, atom_messages: bool = False) -> None:
        """Initialize head.

        Parameters
        ----------
        args : GroverArgs
            Arguments.
        hidden_size : int
            Hidden dimension.
        atom_messages : bool, default=False
            Use atom messages.
        """
        super().__init__()
        atom_fdim = hidden_size
        bond_fdim = hidden_size
        hidden_size = hidden_size
        self.atom_messages = atom_messages
        if self.atom_messages:
            init_message_dim = atom_fdim
            attached_fea_dim = bond_fdim
        else:
            init_message_dim = bond_fdim
            attached_fea_dim = atom_fdim

        self.mpn_q = MPNEncoder(
            args=args,
            atom_messages=atom_messages,
            init_message_dim=init_message_dim,
            attached_fea_fdim=attached_fea_dim,
            hidden_size=hidden_size,
            bias=args.bias,
            depth=args.depth,
            dropout=args.dropout,
            undirected=args.undirected,
            dense=args.dense,
            aggregate_to_atom=False,
            attach_fea=False,
            input_layer="none",
            dynamic_depth="truncnorm",
        )
        self.mpn_k = MPNEncoder(
            args=args,
            atom_messages=atom_messages,
            init_message_dim=init_message_dim,
            attached_fea_fdim=attached_fea_dim,
            hidden_size=hidden_size,
            bias=args.bias,
            depth=args.depth,
            dropout=args.dropout,
            undirected=args.undirected,
            dense=args.dense,
            aggregate_to_atom=False,
            attach_fea=False,
            input_layer="none",
            dynamic_depth="truncnorm",
        )
        self.mpn_v = MPNEncoder(
            args=args,
            atom_messages=atom_messages,
            init_message_dim=init_message_dim,
            attached_fea_fdim=attached_fea_dim,
            hidden_size=hidden_size,
            bias=args.bias,
            depth=args.depth,
            dropout=args.dropout,
            undirected=args.undirected,
            dense=args.dense,
            aggregate_to_atom=False,
            attach_fea=False,
            input_layer="none",
            dynamic_depth="truncnorm",
        )

    def forward(
        self,
        f_atoms: torch.Tensor,
        f_bonds: torch.Tensor,
        a2b: torch.Tensor,
        a2a: torch.Tensor,
        b2a: torch.Tensor,
        b2revb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        f_atoms : torch.Tensor
            Atom features.
        f_bonds : torch.Tensor
            Bond features.
        a2b : torch.Tensor
            Atom to bond mapping.
        a2a : torch.Tensor
            Atom to atom mapping.
        b2a : torch.Tensor
            Bond to atom mapping.
        b2revb : torch.Tensor
            Bond to reverse bond mapping.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Query, key, value tensors.
        """
        if self.atom_messages:
            init_messages = f_atoms
            init_attached_features = f_bonds
            a2nei = a2a
            a2attached = a2b
        else:
            init_messages = f_bonds
            init_attached_features = f_atoms
            a2nei = a2b
            a2attached = a2a

        q = self.mpn_q(
            init_messages=init_messages,
            init_attached_features=init_attached_features,
            a2nei=a2nei,
            a2attached=a2attached,
            b2a=b2a,
            b2revb=b2revb,
        )
        k = self.mpn_k(
            init_messages=init_messages,
            init_attached_features=init_attached_features,
            a2nei=a2nei,
            a2attached=a2attached,
            b2a=b2a,
            b2revb=b2revb,
        )
        v = self.mpn_v(
            init_messages=init_messages,
            init_attached_features=init_attached_features,
            a2nei=a2nei,
            a2attached=a2attached,
            b2a=b2a,
            b2revb=b2revb,
        )
        return q, k, v


class MTBlock(nn.Module):
    """Multi-headed attention block with message passing."""

    def __init__(
        self,
        args,
        num_attn_head: int,
        input_dim: int,
        hidden_size: int,
        activation: str = "ReLU",
        dropout: float = 0.0,
        bias: bool = True,
        atom_messages: bool = False,
        cuda: bool = True,
        res_connection: bool = False,
    ) -> None:
        """Initialize MTBlock.

        Parameters
        ----------
        args : GroverArgs
            Arguments.
        num_attn_head : int
            Number of attention heads.
        input_dim : int
            Input dimension.
        hidden_size : int
            Hidden size.
        activation : str, default="ReLU"
            Activation function.
        dropout : float, default=0.0
            Dropout rate.
        bias : bool, default=True
            Use bias.
        atom_messages : bool, default=False
            Use atom messages.
        cuda : bool, default=True
            Use CUDA.
        res_connection : bool, default=False
            Use residual connection.
        """
        super().__init__()
        self.atom_messages = atom_messages
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList()
        self.input_dim = input_dim
        self.cuda = cuda
        self.res_connection = res_connection
        self.act_func = get_activation_function(activation)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)

        self.W_i = nn.Linear(self.input_dim, self.hidden_size, bias=bias)
        self.attn = MultiHeadedAttention(
            h=num_attn_head,
            d_model=self.hidden_size,
            bias=bias,
            dropout=dropout,
        )
        self.W_o = nn.Linear(self.hidden_size * num_attn_head, self.hidden_size, bias=bias)
        self.sublayer = SublayerConnection(self.hidden_size, dropout)
        for _ in range(num_attn_head):
            self.heads.append(Head(args, hidden_size=hidden_size, atom_messages=atom_messages))

    def forward(
        self, batch: tuple, features_batch: list | None = None
    ) -> tuple[tuple, list | None]:
        """Forward pass.

        Parameters
        ----------
        batch : tuple
            Graph batch from BatchMolGraph.get_components().
        features_batch : list | None, optional
            Additional features.

        Returns
        -------
        tuple[tuple, list | None]
            Updated batch and features.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

        if self.atom_messages:
            if f_atoms.shape[1] != self.hidden_size:
                f_atoms = self.W_i(f_atoms)
                f_atoms = self.dropout_layer(self.layernorm(self.act_func(f_atoms)))
        else:
            if f_bonds.shape[1] != self.hidden_size:
                f_bonds = self.W_i(f_bonds)
                f_bonds = self.dropout_layer(self.layernorm(self.act_func(f_bonds)))

        queries = []
        keys = []
        values = []
        for head in self.heads:
            q, k, v = head(f_atoms, f_bonds, a2b, a2a, b2a, b2revb)
            queries.append(q.unsqueeze(1))
            keys.append(k.unsqueeze(1))
            values.append(v.unsqueeze(1))
        queries = torch.cat(queries, dim=1)
        keys = torch.cat(keys, dim=1)
        values = torch.cat(values, dim=1)

        x_out = self.attn(queries, keys, values)
        x_out = x_out.view(x_out.shape[0], -1)
        x_out = self.W_o(x_out)

        x_in = None
        if self.res_connection:
            if self.atom_messages:
                x_in = f_atoms
            else:
                x_in = f_bonds

        if self.atom_messages:
            f_atoms = self.sublayer(x_in, x_out)
        else:
            f_bonds = self.sublayer(x_in, x_out)

        batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        return batch, features_batch


class GTransEncoder(nn.Module):
    """Graph Transformer encoder with dual-view message passing."""

    def __init__(
        self,
        args,
        hidden_size: int,
        edge_fdim: int,
        node_fdim: int,
        dropout: float = 0.0,
        activation: str = "ReLU",
        num_mt_block: int = 1,
        num_attn_head: int = 4,
        atom_emb_output: Union[bool, str] = False,
        bias: bool = False,
        cuda: bool = True,
        res_connection: bool = False,
    ) -> None:
        """Initialize GTransEncoder.

        Parameters
        ----------
        args : GroverArgs
            Arguments.
        hidden_size : int
            Hidden size.
        edge_fdim : int
            Edge feature dimension.
        node_fdim : int
            Node feature dimension.
        dropout : float, default=0.0
            Dropout rate.
        activation : str, default="ReLU"
            Activation function.
        num_mt_block : int, default=1
            Number of MT blocks.
        num_attn_head : int, default=4
            Number of attention heads.
        atom_emb_output : Union[bool, str], default=False
            Output aggregation mode: True, False, None, "atom", "bond", "both".
        bias : bool, default=False
            Use bias.
        cuda : bool, default=True
            Use CUDA.
        res_connection : bool, default=False
            Use residual connection.
        """
        super().__init__()

        if atom_emb_output is False:
            atom_emb_output = None
        if atom_emb_output is True:
            atom_emb_output = "atom"

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.cuda = cuda
        self.bias = bias
        self.res_connection = res_connection
        self.edge_blocks = nn.ModuleList()
        self.node_blocks = nn.ModuleList()

        edge_input_dim = edge_fdim
        node_input_dim = node_fdim
        edge_input_dim_i = edge_input_dim
        node_input_dim_i = node_input_dim

        for i in range(num_mt_block):
            if i != 0:
                edge_input_dim_i = self.hidden_size
                node_input_dim_i = self.hidden_size
            self.edge_blocks.append(
                MTBlock(
                    args=args,
                    num_attn_head=num_attn_head,
                    input_dim=edge_input_dim_i,
                    hidden_size=self.hidden_size,
                    activation=activation,
                    dropout=dropout,
                    bias=self.bias,
                    atom_messages=False,
                    cuda=cuda,
                )
            )
            self.node_blocks.append(
                MTBlock(
                    args=args,
                    num_attn_head=num_attn_head,
                    input_dim=node_input_dim_i,
                    hidden_size=self.hidden_size,
                    activation=activation,
                    dropout=dropout,
                    bias=self.bias,
                    atom_messages=True,
                    cuda=cuda,
                )
            )

        self.atom_emb_output = atom_emb_output

        self.ffn_atom_from_atom = PositionwiseFeedForward(
            self.hidden_size + node_fdim,
            self.hidden_size * 4,
            activation=self.activation,
            dropout=self.dropout,
            d_out=self.hidden_size,
        )

        self.ffn_atom_from_bond = PositionwiseFeedForward(
            self.hidden_size + node_fdim,
            self.hidden_size * 4,
            activation=self.activation,
            dropout=self.dropout,
            d_out=self.hidden_size,
        )

        self.ffn_bond_from_atom = PositionwiseFeedForward(
            self.hidden_size + edge_fdim,
            self.hidden_size * 4,
            activation=self.activation,
            dropout=self.dropout,
            d_out=self.hidden_size,
        )

        self.ffn_bond_from_bond = PositionwiseFeedForward(
            self.hidden_size + edge_fdim,
            self.hidden_size * 4,
            activation=self.activation,
            dropout=self.dropout,
            d_out=self.hidden_size,
        )

        self.atom_from_atom_sublayer = SublayerConnection(
            size=self.hidden_size, dropout=self.dropout
        )
        self.atom_from_bond_sublayer = SublayerConnection(
            size=self.hidden_size, dropout=self.dropout
        )
        self.bond_from_atom_sublayer = SublayerConnection(
            size=self.hidden_size, dropout=self.dropout
        )
        self.bond_from_bond_sublayer = SublayerConnection(
            size=self.hidden_size, dropout=self.dropout
        )

        self.act_func_node = get_activation_function(self.activation)
        self.act_func_edge = get_activation_function(self.activation)

        self.dropout_layer = nn.Dropout(p=args.dropout)

    def _pointwise_feed_forward_to_atom_embedding(
        self,
        emb_output: torch.Tensor,
        atom_fea: torch.Tensor,
        index: torch.Tensor,
        ffn_layer: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Point-wise feed forward for atom view.

        Parameters
        ----------
        emb_output : torch.Tensor
            Output from multi-head attention.
        atom_fea : torch.Tensor
            Atom features.
        index : torch.Tensor
            Neighborhood index.
        ffn_layer : nn.Module
            FFN layer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output and aggregated output.
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, index)
        aggr_outputx = torch.cat([atom_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    def _pointwise_feed_forward_to_bond_embedding(
        self,
        emb_output: torch.Tensor,
        bond_fea: torch.Tensor,
        a2nei: torch.Tensor,
        b2revb: torch.Tensor,
        ffn_layer: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Point-wise feed forward for bond view.

        Parameters
        ----------
        emb_output : torch.Tensor
            Output from multi-head attention.
        bond_fea : torch.Tensor
            Bond features.
        a2nei : torch.Tensor
            Neighborhood relation.
        b2revb : torch.Tensor
            Reverse bond mapping.
        ffn_layer : nn.Module
            FFN layer.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Output and aggregated output.
        """
        aggr_output = select_neighbor_and_aggregate(emb_output, a2nei)
        aggr_output = self._remove_rev_bond_message(emb_output, aggr_output, b2revb)
        aggr_outputx = torch.cat([bond_fea, aggr_output], dim=1)
        return ffn_layer(aggr_outputx), aggr_output

    @staticmethod
    def _remove_rev_bond_message(
        original_message: torch.Tensor, aggr_message: torch.Tensor, b2revb: torch.Tensor
    ) -> torch.Tensor:
        """Remove reverse bond message.

        Parameters
        ----------
        original_message : torch.Tensor
            Original message.
        aggr_message : torch.Tensor
            Aggregated message.
        b2revb : torch.Tensor
            Reverse bond mapping.

        Returns
        -------
        torch.Tensor
            Message with reverse removed.
        """
        rev_message = original_message[b2revb]
        return aggr_message - rev_message

    def _atom_bond_transform(
        self,
        to_atom: bool = True,
        atomwise_input: torch.Tensor | None = None,
        bondwise_input: torch.Tensor | None = None,
        original_f_atoms: torch.Tensor | None = None,
        original_f_bonds: torch.Tensor | None = None,
        a2a: torch.Tensor | None = None,
        a2b: torch.Tensor | None = None,
        b2a: torch.Tensor | None = None,
        b2revb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform atom/bond embeddings to final output.

        Parameters
        ----------
        to_atom : bool, default=True
            Output atom embeddings (True) or bond embeddings (False).
        atomwise_input : torch.Tensor | None, optional
            Atom input embeddings.
        bondwise_input : torch.Tensor | None, optional
            Bond input embeddings.
        original_f_atoms : torch.Tensor | None, optional
            Original atom features.
        original_f_bonds : torch.Tensor | None, optional
            Original bond features.
        a2a : torch.Tensor | None, optional
            Atom to atom mapping.
        a2b : torch.Tensor | None, optional
            Atom to bond mapping.
        b2a : torch.Tensor | None, optional
            Bond to atom mapping.
        b2revb : torch.Tensor | None, optional
            Bond to reverse bond mapping.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two output tensors depending on transform direction.
        """
        if to_atom:
            atomwise_input, _ = self._pointwise_feed_forward_to_atom_embedding(
                atomwise_input, original_f_atoms, a2a, self.ffn_atom_from_atom
            )
            atom_in_atom_out = self.atom_from_atom_sublayer(None, atomwise_input)
            bondwise_input, _ = self._pointwise_feed_forward_to_atom_embedding(
                bondwise_input, original_f_atoms, a2b, self.ffn_atom_from_bond
            )
            bond_in_atom_out = self.atom_from_bond_sublayer(None, bondwise_input)
            return atom_in_atom_out, bond_in_atom_out
        else:
            atom_list_for_bond = torch.cat([b2a.unsqueeze(dim=1), a2a[b2a]], dim=1)
            atomwise_input, _ = self._pointwise_feed_forward_to_bond_embedding(
                atomwise_input,
                original_f_bonds,
                atom_list_for_bond,
                b2a[b2revb],
                self.ffn_bond_from_atom,
            )
            atom_in_bond_out = self.bond_from_atom_sublayer(None, atomwise_input)
            bond_list_for_bond = a2b[b2a]
            bondwise_input, _ = self._pointwise_feed_forward_to_bond_embedding(
                bondwise_input,
                original_f_bonds,
                bond_list_for_bond,
                b2revb,
                self.ffn_bond_from_bond,
            )
            bond_in_bond_out = self.bond_from_bond_sublayer(None, bondwise_input)
            return atom_in_bond_out, bond_in_bond_out

    def forward(
        self, batch: tuple, features_batch: list | None = None
    ) -> tuple:
        """Forward pass.

        Parameters
        ----------
        batch : tuple
            Graph batch from BatchMolGraph.get_components().
        features_batch : list | None, optional
            Additional features.

        Returns
        -------
        tuple
            Output embeddings.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch
        if self.cuda or next(self.parameters()).is_cuda:
            f_atoms = f_atoms.cuda()
            f_bonds = f_bonds.cuda()
            a2b = a2b.cuda()
            b2a = b2a.cuda()
            b2revb = b2revb.cuda()
            a2a = a2a.cuda()

        node_batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)
        edge_batch = (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a)

        original_f_atoms, original_f_bonds = f_atoms, f_bonds

        for nb in self.node_blocks:
            node_batch, features_batch = nb(node_batch, features_batch)
        for eb in self.edge_blocks:
            edge_batch, features_batch = eb(edge_batch, features_batch)

        atom_output, _, _, _, _, _, _, _ = node_batch
        _, bond_output, _, _, _, _, _, _ = edge_batch

        if self.atom_emb_output is None:
            return atom_output, bond_output

        if self.atom_emb_output == "atom":
            return self._atom_bond_transform(
                to_atom=True,
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb,
            )
        elif self.atom_emb_output == "bond":
            return self._atom_bond_transform(
                to_atom=False,
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb,
            )
        else:
            atom_embeddings = self._atom_bond_transform(
                to_atom=True,
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb,
            )

            bond_embeddings = self._atom_bond_transform(
                to_atom=False,
                atomwise_input=atom_output,
                bondwise_input=bond_output,
                original_f_atoms=original_f_atoms,
                original_f_bonds=original_f_bonds,
                a2a=a2a,
                a2b=a2b,
                b2a=b2a,
                b2revb=b2revb,
            )
            return (
                (atom_embeddings[0], bond_embeddings[0]),
                (atom_embeddings[1], bond_embeddings[1]),
            )