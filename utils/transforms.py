"""Some useful transforms and pre-filters for datasets."""

from collections.abc import Callable, Sequence
from enum import Enum

import torch
import torch_geometric
from rdkit import Chem


def transform_from_qm9_to_gnn(x: torch_geometric.data.Data) -> torch_geometric.data.Data:
    """Transforms a QM9 dataset Data object into the format used in current work.

    The QM9 dataset uses a different atom indexing convention than the GNN
    model. The following mapping is used to convert the atom indices:

    - 1: 9 (Carbon)
    - 2: 12 (Nitrogen)
    - 3: 6 (Oxygen)
    - 4: 8 (Fluorine)
    - 0: 0 (Hydrogen)

    The edge_index is also converted to a torch Tensor of dtype int64.

    The y field is modified to include only the first 18 variables of the
    QM9 dataset, which are the ones used in the GNN model. The smiles field is
    not modified.

    :param x: The input Data object from the QM9 dataset.
    :return: The transformed Data object that is ready to be used with the GNN model.
    """
    device = x.x.device

    qm9_to_gnn_atom_idxs = {
        1: 9,  # C
        2: 12,  # N
        3: 6,  # O
        4: 8,  # F
        0: 0,  # H
    }

    atom_idx_old = torch.argmax(x.x[:, :5], dim=1)

    edge_index = x.edge_index
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.to(dtype=torch.int64, device=device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.int64, device=device)

    return torch_geometric.data.Data(
        x=torch.tensor(
            [qm9_to_gnn_atom_idxs[cur.item()] for cur in atom_idx_old],
            dtype=torch.int64,
            device=device,
        ),
        edge_index=edge_index,
        y=x.y.squeeze(0)[:-3],
        smiles=x.smiles,
        z=x.z,
    )


def transform_from_odb_to_gnn(
    x: torch_geometric.data.Data,
) -> torch_geometric.data.Data:
    """ODB format -> format used in the current work.

    Transform datum from format with additional node and edge features
    from [OpenGraphBenchmark](https://ogb.stanford.edu/docs/graphprop/)
    to the format used in the current work.
    """
    device = x.x.device

    odb_to_gnn_atom_idxs = {
        1: 0,  # H
        32: 1,  # Ge
        52: 2,  # Te
        15: 3,  # P
        83: 4,  # Bi
        14: 5,  # Si
        8: 6,  # O
        17: 7,  # Cl
        9: 8,  # F
        6: 9,  # C
        16: 10,  # S
        51: 11,  # Sb
        7: 12,  # N
        35: 13,  # Br
        34: 14,  # Se
        5: 15,  # B
        53: 16,  # I
        50: 17,  # Sn
    }

    atom_idx_old = x.x[:, 0]

    edge_index = x.edge_index
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.to(dtype=torch.int64, device=device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.int64, device=device)

    return torch_geometric.data.Data(
        x=torch.tensor(
            [odb_to_gnn_atom_idxs[cur.item()] for cur in atom_idx_old],
            dtype=torch.int64,
            device=device,
        ),
        edge_index=edge_index,
        y=x.y,
        z=atom_idx_old,
        smiles=x.smiles,
    )


def pre_filter_carbon_containing(x: torch_geometric.data.Data) -> bool:
    """Pre-filter function for dataset that filters out molecules with less than one carbon atom.

    Args:
        x: A torch_geometric.data.Data object representing a molecule
            , with atomic number information in `x.z`.

    Returns:
        bool: True if the molecule contains more than one carbon atom, False otherwise.

    """
    # Umm, hy were we doing this, actually? I mean, why >1 carbon, not >0?
    return (x.z != 1).sum().item() > 1


class TransformsSequence:
    """A sequence of transforms to be applied to a torch_geometric.data.Data object."""

    def __init__(
        self,
        transforms: list[Callable[[torch_geometric.data.Data], torch_geometric.data.Data]],
    ) -> None:
        """Initialize a TransformsSequence object.

        Args:
            transforms (list[Callable[[torch_geometric.data.Data],torch_geometric.data.Data]]):
                A list of transforms to be applied in a sequence.

        """
        self.transforms = transforms

    def __call__(self, x: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply the sequence of transforms to the input data.

        Args:
            x (torch_geometric.data.Data): The input data to be transformed.

        Returns:
            torch_geometric.data.Data: The transformed data.

        """
        for transform in self.transforms:
            x = transform(x)
        return x


class AllPreFilters:
    """A set of pre-filters. Allows the item if all pre-filters return True."""

    def __init__(
        self,
        pre_filters: list[Callable[[torch_geometric.data.Data], bool]],
    ) -> None:
        """Initialize a PreFilterSequence object.

        Args:
            pre_filters (list[Callable[[torch_geometric.data.Data],bool]]): A list of pre-filters
            to be applied in a sequence.

        """
        self.pre_filters = pre_filters

    def __call__(self, x: torch_geometric.data.Data) -> bool:
        """Apply the sequence of pre-filters to the input data.

        Args:
            x (torch_geometric.data.Data): The input data to be filtered.

        Returns:
            bool: True if all pre-filters return True, False otherwise.

        """
        return all(pre_filter(x) for pre_filter in self.pre_filters)


class AnyPreFilters:
    """A set of pre-filters. Allows the item if any pre-filters return True."""

    def __init__(
        self,
        pre_filters: list[Callable[[torch_geometric.data.Data], bool]],
    ) -> None:
        """Initialize a PreFilterSequence object.

        Args:
            pre_filters (list[Callable[[torch_geometric.data.Data],bool]]): A list of pre-filters
            to be applied in a sequence.

        """
        self.pre_filters = pre_filters

    def __call__(self, x: torch_geometric.data.Data) -> bool:
        """Apply the sequence of pre-filters to the input data.

        Args:
            x (torch_geometric.data.Data): The input data to be filtered.

        Returns:
            bool: True if any pre-filters return True, False otherwise.

        """
        return any(pre_filter(x) for pre_filter in self.pre_filters)


class PreFilterRemoveLongSMILES:
    """Pre-filter to remove SMILES strings that are longer than a specified length."""

    max_len: int

    def __init__(self, max_len: int) -> None:
        """Initialize a PreFilterRemoveLongSMILES object.

        Args:
            max_len (int): The maximum length of a SMILES string.

        """
        self.max_len = max_len

    def __call__(self, x: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply the pre-filter to remove long SMILES strings.

        Args:
            x (torch_geometric.data.Data): The input data to be filtered.

        Returns:
            True if the SMILES string is less than or equal to the maximum length, False otherwise.

        """
        return len(x.smiles) <= self.max_len


class AtomicNumberDeterminationFormat(Enum):
    """Atomic number determination format."""

    FIRST_ELEMENT_OF_X = 0
    Z = "z"


class PreFilterOnlyTheseAtoms:
    """Pre-filter to remove molecules that don't contain only the specified atoms."""

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        determination_format: AtomicNumberDeterminationFormat = AtomicNumberDeterminationFormat.FIRST_ELEMENT_OF_X,  # noqa: E501
    ) -> None:
        """Initialize a PreFilterOnlyTheseAtoms object.

        Args:
            atomic_numbers (Sequence[int]): A sequence of atomic numbers.
            determination_format (AtomicNumberDeterminationFormat): The format to determine the
                atomic numbers of the molecule.

        """
        self.atomic_numbers = set(atomic_numbers)
        self.determination_format = determination_format

    def __call__(self, x: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Apply the pre-filter to remove molecules that don't contain only the specified atoms.

        Args:
            x (torch_geometric.data.Data): The input data to be filtered.

        Returns:
            True if the molecule contains only the specified atoms, False otherwise.

        """
        if self.determination_format == AtomicNumberDeterminationFormat.FIRST_ELEMENT_OF_X:
            return self._use_first_element_of_x(x)
        if self.determination_format == AtomicNumberDeterminationFormat.Z:
            return self._use_z(x)
        raise ValueError(self.determination_format)

    def _use_z(self, x: torch_geometric.data.Data) -> bool:
        """Use the z field of the data to determine the atomic numbers of the molecule.

        Returns:
            True if the molecule contains only the specified atoms, False otherwise.

        """
        return all(cur.item() in self.atomic_numbers for cur in x.z)

    def _use_first_element_of_x(self, x: torch_geometric.data.Data) -> bool:
        """Use the first element of the x field of the data to determine the atomic numbers.

        Returns:
            True if the molecule contains only the specified atoms, False otherwise.

        """
        return all(cur.item() in self.atomic_numbers for cur in x.x[:, 0])


class PreFilterCheckValidSMILES:
    """Pre-filter to check if the SMILES string is valid.

    Returns:
        True if the SMILES string is valid, False otherwise.

    """

    def __call__(self, x: torch_geometric.data.Data) -> torch_geometric.data.Data:
        """Check if the SMILES string is valid.

        Args:
            x (torch_geometric.data.Data): The input data to be checked.

        Returns:
            bool: True if the SMILES string is valid, False otherwise.

        """
        mol = Chem.MolFromSmiles(x.smiles)
        return mol is not None
