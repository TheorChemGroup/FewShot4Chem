"""Minor utilities."""

import re
from collections.abc import Sequence
from pathlib import Path

import lightning as L  # noqa: N812
import torch
import torch_geometric
from plotly.graph_objects import Figure
from torch import nn

from utils.typing import DataItem

UNKNOWN_DATA_TYPE = "Unknown data type: {data_type}"


class MeanOneMinusCosineSimilarity(nn.Module):
    """Module that computes the mean of 1 - cosine similarity between two tensors.

    This module can be used as a loss function for training models that require
    cosine similarity between their outputs and the target values.

    Parameters
    ----------
    dim : int, optional
        Dimension of the cosine similarity, by default 1.
    eps : float, optional
        Small value to prevent division by zero, by default 1e-8.

    Attributes
    ----------
    cosine_similarity : nn.CosineSimilarity
        Module that computes the cosine similarity between two tensors.

    Methods
    -------
    forward(x, y)
        Compute the mean of 1 - cosine similarity between x and y.

    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Initialize the MeanOneMinusCosineSimilarity module.

        Parameters
        ----------
        dim : int, optional
            Dimension of the cosine similarity, by default 1.
        eps : float, optional
            Small value to prevent division by zero, by default 1e-8.

        """
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim, eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the mean of 1 - cosine similarity between x and y.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mean of 1 - cosine similarity between x and y.

        """
        return 1 - self.cosine_similarity(x, y).mean()


def safe_write_image(fig: Figure, path: str | Path) -> None:
    """Safely writes a plotly figure to a file.

    Parameters
    ----------
    fig : Figure
        The plotly figure to write.
    path : str | Path
        The path to write the figure to.

    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path)


def get_smiles(mol: DataItem, dataset: Sequence[DataItem]) -> str:
    """Get the SMILES of a molecule from a dataset.

    Parameters
    ----------
    mol : DataItem
        The molecule to get the SMILES of.
    dataset : Sequence[DataItem]
        The dataset from which the molecule was taken. This is used to understand
        the format of the molecule and get tokenizer if needed.

    Returns
    -------
    str
        The SMILES of the molecule.

    """
    try:
        return mol.smiles
    except AttributeError:
        tokenizer = dataset.tokenizer
        return re.sub(" ", "", tokenizer.decode(mol["ids"].detach().cpu().numpy()))


def put_on_device(
    model: nn.Module | L.LightningModule,
    data: torch.Tensor | torch_geometric.data.Data | dict | list,
) -> torch.Tensor | torch_geometric.data.Data:
    """Move the data to the device of the model.

    Recursively delves into lists and dictionaries to move the data to the device of the model.

    Parameters
    ----------
    model : nn.Module | L.LightningModule
        The model which device to use.
    data : torch.Tensor | torch_geometric.data.Data | dict | list
        The data to move to the device.

    Returns
    -------
    torch.Tensor | torch_geometric.data.Data
        The data moved to the device of the model.

    """
    if isinstance(data, (torch_geometric.data.Data, torch.Tensor)):
        data_on_device = data.to(model.device)
    elif isinstance(data, dict):
        data_on_device = {
            k: put_on_device(model, v) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
    elif isinstance(data, list):
        data_on_device = [put_on_device(model, v) for v in data]
    else:
        raise TypeError(UNKNOWN_DATA_TYPE.format(data_type=type(data)))

    return data_on_device
