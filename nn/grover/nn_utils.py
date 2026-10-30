"""Utility functions for GROVER model construction.

Adapted from https://github.com/tencent-ailab/grover
"""

import torch
from torch import nn


def param_count(model: nn.Module) -> int:
    """Determines number of trainable parameters.

    Parameters
    ----------
    model : nn.Module
        An nn.Module.

    Returns
    -------
    int
        The number of trainable parameters.

    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def index_select_nd(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Selects features from source corresponding to indices.

    Parameters
    ----------
    source : torch.Tensor
        A tensor of shape (num_items, hidden_size) containing features.
    index : torch.Tensor
        A tensor of shape (num_items, max_num_neighbors) containing indices.

    Returns
    -------
    torch.Tensor
        A tensor of shape (num_items, max_num_neighbors, hidden_size).

    """
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim

    target = source.index_select(dim=0, index=index.view(-1))
    target = target.view(final_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    """Gets an activation function module given the name.

    Parameters
    ----------
    activation : str
        The name of the activation function.

    Returns
    -------
    nn.Module
        The activation function module.

    Raises
    ------
    ValueError
        If activation is not supported.

    """
    if activation == "ReLU":
        return nn.ReLU()
    if activation == "LeakyReLU":
        return nn.LeakyReLU(0.1)
    if activation == "PReLU":
        return nn.PReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "SELU":
        return nn.SELU()
    if activation == "ELU":
        return nn.ELU()
    if activation == "Linear":

        def linear(x):
            return x

        return linear
    raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Module, distinct_init: bool = False, model_idx: int = 0) -> None:
    """Initializes the weights of a model in place.

    Parameters
    ----------
    model : nn.Module
        An nn.Module.
    distinct_init : bool, default=False
        Whether to use distinct initialization for different models.
    model_idx : int, default=0
        Model index for distinct initialization.

    """
    init_fns = [
        nn.init.kaiming_normal_,
        nn.init.kaiming_uniform_,
        nn.init.xavier_normal_,
        nn.init.xavier_uniform_,
    ]
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        elif distinct_init:
            init_fn = init_fns[model_idx % 4]
            if "kaiming" in init_fn.__name__:
                init_fn(param, nonlinearity="relu")
            else:
                init_fn(param)
        else:
            nn.init.xavier_normal_(param)


def select_neighbor_and_aggregate(feature: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The basic operation in message passing.

    Caution: the index_select_nd would cause the reproducibility issue
    when performing the training on CUDA.
    See: https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    feature : torch.Tensor
        The candidate feature for aggregate. (n_nodes, hidden)
    index : torch.Tensor
        The selected index (neighbor indexes).

    Returns
    -------
    torch.Tensor
        Aggregated neighbor features.

    """
    neighbor = index_select_nd(feature, index)
    return neighbor.sum(dim=1)
