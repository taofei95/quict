import os.path as osp
from os import walk
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset as PygDataset, Data as PygData
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch as PygBatch
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.nn.models import Node2Vec
from typing import List, Set, Tuple, Iterable
from QuICT.core import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from QuICT.tools.interface import OPENQASMInterface
import os
from random import shuffle
from QuICT.qcda.mapping.ai.data_def import PairData
from QuICT.qcda.mapping.ai.data_processor_graphormer import (
    CircuitGraphormerDataProcessor,
)
import networkx as nx


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


class MappingGraphormerDataset:
    def __init__(self, data_dir: str = None) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
        processed_dir = osp.join(data_dir, "processed_graphormer")
        self._processed_dir = processed_dir
        topo_dir = osp.join(data_dir, "topo")

        metadata_path = osp.join(processed_dir, "metadata.pt")
        self.metadata = torch.load(metadata_path)  # (max_qubit_num, max_layer_num)

        self._processor = CircuitGraphormerDataProcessor(
            max_qubit_num=self.metadata[0],
            max_layer_num=self.metadata[1],
            data_dir=data_dir,
        )

        self._topo_graph_map = {}
        for topo_name in self._processor.get_topo_names():
            topo_path = osp.join(topo_dir, f"{topo_name}.layout")
            topo = Layout.load_file(topo_path)
            topo_graph = self._processor.get_topo_graph(topo)
            self._topo_graph_map[topo_name] = topo_graph

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.IntTensor]:
        f_path = osp.join(self._processed_dir, f"{idx}.pt")
        with open(f_path, "rb") as f:
            layered_circ, topo_name = torch.load(f)
            topo_graph = self._topo_graph_map[topo_name]
            x = self._processor.get_x()
            circ_graph = self._processor.get_circ_graph(layered_circ=layered_circ)
            spacial_encoding = self._processor.get_spacial_encoding(
                circ_graph=circ_graph, topo_graph=topo_graph
            )
            return x, spacial_encoding


if __name__ == "__main__":
    # This is used as test. It's not suitable to be put into unit_test because it depends on 
    # data we collected.
    dataset = MappingGraphormerDataset()
    qubit_num, layer_num = dataset.metadata
    node_num = qubit_num * layer_num + 1
    x, spacial_encoding = dataset[0]

    assert x.shape == torch.Size((node_num,))
    assert spacial_encoding.shape == torch.Size((node_num, node_num,))
