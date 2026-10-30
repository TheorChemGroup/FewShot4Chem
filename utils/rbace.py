import re
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import from_smiles as _from_smiles


class RBACE(InMemoryDataset):
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        *,
        force_reload: bool = False,
        from_smiles: Callable | None = None,
    ) -> None:
        self.from_smiles = from_smiles or _from_smiles
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return Path(self.root) / "raw"

    @property
    def processed_dir(self) -> str:
        return Path(self.root) / "processed"

    @property
    def raw_file_names(self) -> str:
        return "bace.csv"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        download_url(self.url, self.raw_dir)

    def process(self) -> None:
        with Path(self.raw_paths[0]).open(encoding="utf-8") as f:
            dataset = f.read().split("\n")[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line_raw in dataset:
            line = re.sub(r"\".*\"", "", line_raw)  # Replace ".*" strings.
            values = line.split(",")

            smiles = values[0]
            labels = values[4]
            labels = labels if isinstance(labels, list) else [labels]

            ys = [float(y) if len(y) > 0 else float("NaN") for y in labels]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            data = self.from_smiles(smiles)
            data.y = y

            if data.num_nodes == 0:
                warnings.warn(
                    f"Skipping molecule '{smiles}' since it resulted in zero atoms",
                    stacklevel=2,
                )
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f"RBACE({len(self)})"
