"""GIN architecture matching Hu et al. ICLR 2020 pretrained checkpoints.

Verbatim port from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py
of GINConv + GNN. The custom GINConv adds bond embeddings INSIDE the message function;
loading checkpoints with mismatched architecture silently produces wrong embeddings.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

NUM_ATOM_TYPE = 120
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3


class GINConv(MessagePassing):
    """Hu et al. GIN layer: bond embeddings added inside message function."""

    def __init__(self, emb_dim: int, aggr: str = "add") -> None:
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros((x.size(0), 2), dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return x_j + edge_attr

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.mlp(aggr_out)


class GNN(nn.Module):
    """Hu et al. GNN backbone: atom-feature embeddings + 5 GIN layers + global mean pool."""

    def __init__(self, num_layer: int = 5, emb_dim: int = 300, drop_ratio: float = 0.0) -> None:
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer < self.num_layer - 1:
                h = nn.functional.relu(h)
            h = nn.functional.dropout(h, self.drop_ratio, training=self.training)
        return h
