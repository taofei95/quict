from os import path as osp
import os
from typing import Dict, List, Tuple
from torch_geometric.data import HeteroData, Data, Batch
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

from QuICT.qcda.mapping.ai.data_def import PairData
from QuICT.qcda.mapping.ai.data_processor import MappingDataProcessor


class MappingDataSet(Dataset):
    def __init__(self, data_dir: str, device: str) -> None:
        super().__init__()
        self._file_names = []
        self._data_dir = data_dir
        self.device = device
        if not osp.exists(data_dir):
            raise FileNotFoundError(data_dir)
        print(f"Loading dataset from {data_dir}")
        for _, _, filenames in os.walk(data_dir):
            self._file_names = filenames

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, idx: int) -> Tuple[PairData, int]:
        f_path = osp.join(self._data_dir, self._file_names[idx])
        with open(f_path, "rb") as f:
            circ, target = torch.load(f)
            return circ, target


class MappingDataLoaderFactory:
    @staticmethod
    def get_loader(
        batch_size: int, shuffle: bool, device: str, data_dir: str = None,
    ):
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed")

        dataset = MappingDataSet(data_dir=data_dir, device=device)

        def collat_fn(batch):
            circuits, targets = zip(*batch)
            assert len(batch) == len(circuits) and len(batch) == len(targets)
            circuits = Batch.from_data_list(
                list(circuits), follow_batch=["x_topo", "x_lc"]
            ).to(device=device)
            targets = (
                torch.tensor(list(targets), dtype=torch.float)
                .unsqueeze(dim=1)
                .to(device=device)
            )
            return circuits, targets

        loader = TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collat_fn,
        )
        return loader


if __name__ == "__main__":
    loader = MappingDataLoaderFactory.get_loader(
        batch_size=10, shuffle=True, device="cpu"
    )
    for batch in loader:
        print(batch)
        break
