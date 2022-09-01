import os.path as osp
from os import walk
import torch
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.data import HeteroData
from typing import Tuple, Iterable
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
   
