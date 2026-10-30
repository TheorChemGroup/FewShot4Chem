import torch
import torch_geometric
from rdkit.Chem import MolFromSmiles, rdMolDescriptors

from nn._register_model import register_model_type
from nn.multimodal_module import MultiModalModule


@register_model_type("descriptors")
class LightningDescriptors(MultiModalModule, modes=("embedding",)):
    """LightningDescriptors is a LightningModule that wraps RDKit descriptors."""

    def __init__(self, mode: str = "embedding") -> None:
        super().__init__(mode=mode)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """Forward method for LightningDescriptors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing a "smiles" field.

        Returns
        -------
        torch.Tensor
            Descriptors for the input SMILES.

        """
        smiles_list = data.smiles
        props = rdMolDescriptors.Properties()
        all_descriptors_list = []
        for smiles in smiles_list:
            mol = MolFromSmiles(smiles)
            descriptors = props.ComputeProperties(mol)
            descriptors_list = list(descriptors)
            all_descriptors_list.append(descriptors_list)
        return torch.tensor(all_descriptors_list, device=data.x.device)
