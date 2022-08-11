import os.path as osp
from os import walk
import torch
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.data import Batch as PygBatch
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn.models import Node2Vec
from typing import Tuple, Iterable
from QuICT.core import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from QuICT.tools.interface import OPENQASMInterface

from QuICT.qcda.mapping.ai.data_def import PairData


class MappingDataset(PygDataset):
    def __init__(self, data_dir: str = None) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed")
            
        super().__init__()
        self._file_names = []
        self._data_dir = data_dir
        if not osp.exists(data_dir):
            raise FileNotFoundError(data_dir)
        print(f"Loading dataset from {data_dir}")
        for _, _, filenames in walk(data_dir):
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
    def get_loader(batch_size: int, shuffle: bool, device, data_dir: str = None):
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed")

        dataset = MappingDataset(data_dir=data_dir)

        loader = PygDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            follow_batch=["x_topo", "x_lc"],
        )
        return loader


if __name__ == "__main__":
    loader = MappingDataLoaderFactory.get_loader(batch_size=10, shuffle=True)
    cnt = 0
    print(len(loader))
    for batch in loader:
        print(batch)
        cnt += 1
        if cnt == 3:
            break
