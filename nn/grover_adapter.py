"""Adapter to convert PyTorch Geometric data to GROVER format."""


import torch
import torch_geometric

from nn.grover.molgraph import BatchMolGraph, GroverArgs, MolGraph


def pyg_to_grover_batch(
    data: torch_geometric.data.Data | torch_geometric.data.Batch,
    args: GroverArgs | None = None,
) -> BatchMolGraph:
    """Convert PyG Data/Batch to GROVER BatchMolGraph.

    Parameters
    ----------
    data : torch_geometric.data.Data | torch_geometric.data.Batch
        PyG data object containing 'smiles' attribute.
    args : GroverArgs | None, optional
        GROVER arguments for graph construction.

    Returns
    -------
    BatchMolGraph
        GROVER batch molecular graph.

    Raises
    ------
    ValueError
        If 'smiles' attribute is missing from data.

    """
    if args is None:
        args = GroverArgs()

    if not hasattr(data, "smiles"):
        raise ValueError("Data object must have 'smiles' attribute for GROVER conversion")

    smiles_attr = data.smiles

    if isinstance(smiles_attr, str):
        smiles_list = [smiles_attr]
    elif isinstance(smiles_attr, list):
        smiles_list = smiles_attr
    elif isinstance(smiles_attr, torch.Tensor):
        smiles_list = smiles_attr.tolist()
    else:
        smiles_list = list(smiles_attr)

    mol_graphs = []
    for smiles in smiles_list:
        if smiles is not None and isinstance(smiles, str):
            mol_graph = MolGraph(smiles, args)
            if mol_graph.n_atoms > 0:
                mol_graphs.append(mol_graph)

    if not mol_graphs:
        dummy_mol = MolGraph("C", args)
        mol_graphs = [dummy_mol]

    return BatchMolGraph(mol_graphs, args)


def pyg_batch_to_grover_batch(
    batch: torch_geometric.data.Batch,
    args: GroverArgs | None = None,
) -> tuple:
    """Convert PyG Batch to GROVER batch tuple.

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        PyG batch object containing 'smiles' attribute.
    args : GroverArgs | None, optional
        GROVER arguments for graph construction.

    Returns
    -------
    tuple
        GROVER batch tuple from BatchMolGraph.get_components().

    """
    grover_batch = pyg_to_grover_batch(batch, args)
    return grover_batch.get_components()


class PyGToGroverAdapter:
    """Adapter class for converting PyG data to GROVER format.

    This adapter handles the conversion between PyTorch Geometric Data objects
    and GROVER's BatchMolGraph format, enabling GROVER to work with PyG pipelines.

    Attributes
    ----------
    args : GroverArgs
        GROVER configuration arguments.

    """

    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.1,
        activation: str = "PReLU",
        depth: int = 3,
        num_mt_block: int = 1,
        num_attn_head: int = 4,
        bias: bool = False,
        undirected: bool = True,
        dense: bool = True,
        cuda: bool = False,
        bond_drop_rate: float = 0.0,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        hidden_size : int, default=128
            Hidden size for GROVER model.
        dropout : float, default=0.1
            Dropout rate.
        activation : str, default="PReLU"
            Activation function.
        depth : int, default=3
            Message passing depth.
        num_mt_block : int, default=1
            Number of multi-head attention blocks.
        num_attn_head : int, default=4
            Number of attention heads.
        bias : bool, default=False
            Use bias in linear layers.
        undirected : bool, default=True
            Use undirected message passing.
        dense : bool, default=True
            Use dense connections.
        cuda : bool, default=False
            Use CUDA.
        bond_drop_rate : float, default=0.0
            Rate of dropping bonds during training.

        """
        self.args = GroverArgs(
            hidden_size=hidden_size,
            dropout=dropout,
            activation=activation,
            depth=depth,
            num_mt_block=num_mt_block,
            num_attn_head=num_attn_head,
            bias=bias,
            undirected=undirected,
            dense=dense,
            cuda=cuda,
            bond_drop_rate=bond_drop_rate,
        )

    def convert(self, data: torch_geometric.data.Data | torch_geometric.data.Batch) -> tuple:
        """Convert PyG data to GROVER batch tuple.

        Parameters
        ----------
        data : torch_geometric.data.Data | torch_geometric.data.Batch
            PyG data object with 'smiles' attribute.

        Returns
        -------
        tuple
            GROVER batch tuple.

        """
        return pyg_batch_to_grover_batch(data, self.args)

    def convert_to_batch_mol_graph(
        self, data: torch_geometric.data.Data | torch_geometric.data.Batch,
    ) -> BatchMolGraph:
        """Convert PyG data to GROVER BatchMolGraph.

        Parameters
        ----------
        data : torch_geometric.data.Data | torch_geometric.data.Batch
            PyG data object with 'smiles' attribute.

        Returns
        -------
        BatchMolGraph
            GROVER batch molecular graph.

        """
        return pyg_to_grover_batch(data, self.args)

    def get_args(self) -> GroverArgs:
        """Get the GROVER arguments.

        Returns
        -------
        GroverArgs
            The configuration arguments.

        """
        return self.args


def extract_smiles_from_data(
    data: torch_geometric.data.Data | torch_geometric.data.Batch,
) -> list[str]:
    """Extract SMILES strings from PyG data object.

    Parameters
    ----------
    data : torch_geometric.data.Data | torch_geometric.data.Batch
        PyG data object.

    Returns
    -------
    list[str]
        List of SMILES strings.

    """
    if not hasattr(data, "smiles"):
        return []

    smiles_attr = data.smiles

    if isinstance(smiles_attr, str):
        return [smiles_attr]
    if isinstance(smiles_attr, list):
        return [s for s in smiles_attr if s is not None]
    if isinstance(smiles_attr, torch.Tensor):
        return [s for s in smiles_attr.tolist() if s is not None]
    return [s for s in list(smiles_attr) if s is not None]
