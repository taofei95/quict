import os.path as osp
from os import walk
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset as PygDataset, Data as PygData
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch as PygBatch
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn.models import Node2Vec
from typing import List, Tuple, Iterable
from QuICT.core import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from QuICT.tools.interface import OPENQASMInterface
import os
from random import shuffle
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


class MappingBaseDataset(PygDataset):
    def __init__(self, data_dir: str = None, file_names: Iterable[str] = None) -> None:
        super().__init__()
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed")

        super().__init__()

        self._data_dir = data_dir
        if file_names is None:
            self._file_names = []
            if not osp.exists(data_dir):
                raise FileNotFoundError(data_dir)
            print(f"Loading dataset from {data_dir}")
            for _, _, filenames in os.walk(data_dir):
                self._file_names = filenames
        else:
            self._file_names = file_names

    def __len__(self):
        return len(self._file_names)


class MappingHeteroDataset(MappingBaseDataset):
    def __init__(self, data_dir: str = None, file_names: Iterable[str] = None) -> None:
        super().__init__(data_dir=data_dir, file_names=file_names)

    def split_tv(
        self, point: int = 90
    ) -> Tuple["MappingHeteroDataset", "MappingHeteroDataset"]:
        shuffle(self._file_names)
        p = len(self) * point // 100
        return (
            MappingHeteroDataset(
                data_dir=self._data_dir, file_names=self._file_names[:p]
            ),
            MappingHeteroDataset(
                data_dir=self._data_dir, file_names=self._file_names[p:]
            ),
        )

    @classmethod
    def to_hetero_data(cls, raw_data: PairData) -> HeteroData:
        data = HeteroData()
        data["topo"].x = raw_data.x_topo
        data["lc"].x = raw_data.x_lc

        data["topo", "topo_edge", "topo"].edge_index = raw_data.edge_index_topo
        data["lc", "lc_edge", "lc"].edge_index = raw_data.edge_index_lc

        return data

    def __getitem__(self, idx: int) -> Tuple[HeteroData, int]:
        f_path = osp.join(self._data_dir, self._file_names[idx])
        with open(f_path, "rb") as f:
            raw_data, target = torch.load(f)
            data = self.to_hetero_data(raw_data)
            return data, target


class MappingGraphormerDataset(MappingBaseDataset):
    def __init__(self, data_dir: str = None, file_names: Iterable[str] = None) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
            data_dir = osp.join(data_dir, "processed_graphormer")
        super().__init__(data_dir=data_dir, file_names=file_names)

    def split_tv(
        self, point: int = 90
    ) -> Tuple["MappingGraphormerDataset", "MappingGraphormerDataset"]:
        shuffle(self._file_names)
        p = len(self) * point // 100
        return (
            MappingGraphormerDataset(
                data_dir=self._data_dir, file_names=self._file_names[:p]
            ),
            MappingGraphormerDataset(
                data_dir=self._data_dir, file_names=self._file_names[p:]
            ),
        )

    def __getitem__(self, idx: int):
        f_path = osp.join(self._data_dir, self._file_names[idx])
        with open(f_path, "rb") as f:
            x, spacial_encoding = torch.load(f)
            # Some process
            return x, spacial_encoding

    def loader(self, batch_size: int, shuffle: bool, device: str = "cpu"):
        def _collate_fn(data_list):
            xs, spacial_encodings = zip(*data_list)
            xs = torch.stack(xs).to(device=device)
            spacial_encodings = torch.stack(spacial_encodings).to(device=device)
            return xs, spacial_encodings

        return DataLoader(
            dataset=self, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn
        )
