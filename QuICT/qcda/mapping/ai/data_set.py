from os import path as osp
import os
from typing import List, Tuple
from torch_geometric.data import HeteroData, Data, Batch
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset


class MappingDataSet(Dataset):
    def __init__(self, data_dir: str, device: str) -> None:
        super().__init__()
        self._circ_cnt = 0
        self._data_dir = data_dir
        self.device = device
        print(f"Loading dataset from {data_dir}")
        for _, _, filenames in os.walk(data_dir):
            self._circ_cnt += len(filenames)

    def __len__(self):
        return self._circ_cnt

    def __getitem__(self, idx: int) -> Tuple[Data, Data, int]:
        f_path = osp.join(self._data_dir, f"{idx}.pt")
        with open(f_path, "rb") as f:
            pc_conn, lc_circ, target = torch.load(f)
            pc_conn = pc_conn.to(self.device)
            lc_circ = lc_circ.to(self.device)
            return pc_conn, lc_circ, target


class MappingDataLoaderFactory:
    @staticmethod
    def get_loader(
        data_dir: str = None,
        batch_size: int = 32,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed")

        dataset = MappingDataSet(data_dir=data_dir, device=device)

        def collat_fn(batch):
            pc_connections, lc_circuits, targets = zip(*batch)
            pc_connections = list(pc_connections)
            lc_circuits = list(lc_circuits)
            targets = list(targets)
            pc_connections = Batch.from_data_list(pc_connections)
            lc_circuits = Batch.from_data_list(lc_circuits)
            targets = torch.tensor(targets, dtype=torch.long).to(device=device)
            return pc_connections, lc_circuits, targets

        loader = TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collat_fn,
        )
        return loader


if __name__ == "__main__":
    loader = MappingDataLoaderFactory.get_loader(batch_size=1)
    for batch in loader:
        print(batch)
        break
