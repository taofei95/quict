import networkx as nx
from random import choice, randint
import os.path as osp
from QuICT.qcda.mapping.ai.data_processor_transformer import (
    CircuitTransformerDataProcessor,
)
from typing import Dict, List, Tuple, Set
import torch
from QuICT.core import *
from QuICT.core.gate import GateType


class CircuitTransformerDataFactory:
    def __init__(
        self, max_qubit_num: int, max_layer_num: int, data_dir: str = None
    ) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
        self._topo_dir = osp.join(data_dir, "topo")

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num

        self.processor = CircuitTransformerDataProcessor(
            max_qubit_num=max_qubit_num,
            max_layer_num=max_layer_num,
            data_dir=data_dir,
        )

        # Topo attr cache def
        # These attributes maps are lazily initialized for faster start up.

        self._topo_graph_map = {}
        self._topo_edge_map = {}
        self._topo_qubit_num_map = {}
        self._topo_x_map = {}
        self._topo_mask_map = {}
        self._topo_dist_map = {}

    @property
    def topo_graph_map(self) -> Dict[str, nx.Graph]:
        if len(self._topo_graph_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_graph_map

    @property
    def topo_edge_map(self) -> Dict[str, List[Tuple[int, int]]]:
        if len(self._topo_edge_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_edge_map

    @property
    def topo_qubit_num_map(self) -> Dict[str, int]:
        if len(self._topo_qubit_num_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_qubit_num_map

    @property
    def topo_x_map(self) -> Dict[str, torch.IntTensor]:
        if len(self._topo_x_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_x_map

    @property
    def topo_mask_map(self) -> Dict[str, torch.Tensor]:
        if len(self._topo_mask_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_mask_map

    @property
    def topo_dist_map(self) -> Dict[str, torch.IntTensor]:
        if len(self._topo_dist_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_dist_map

    def _reset_topo_attr_cache(self):
        for topo_name in self.processor.topo_names:
            topo_path = osp.join(self._topo_dir, f"{topo_name}.layout")
            topo = Layout.load_file(topo_path)
            topo_graph = self.processor.get_topo_graph(topo)
            self._topo_graph_map[topo_name] = topo_graph
            self._topo_qubit_num_map[topo_name] = topo.qubit_number
            self._topo_x_map[topo_name] = self.processor.get_x(topo.qubit_number)
            self._topo_dist_map[topo_name] = self.processor.get_topo_dist(
                topo_graph=topo_graph
            )

            topo_mask = torch.zeros(
                (topo.qubit_number, topo.qubit_number), dtype=torch.float
            )
            topo_edge = []
            for u, v in topo_graph.edges:
                topo_mask[u][v] = 1.0
                topo_mask[v][u] = 1.0
                topo_edge.append((u, v))
                topo_edge.append((v, u))
            self._topo_mask_map[topo_name] = topo_mask
            self._topo_edge_map = topo_edge

    def get_one(
        self,
    ) -> Tuple[
        List[Set[Tuple[int, int]]], str, torch.IntTensor, torch.IntTensor, List[int]
    ]:
        """
        Returns:
            Tuple[List[Set[Tuple[int, int]]], str, torch.IntTensor, torch.IntTensor]: (layered_circ, topo_name, x, spacial_encoding, cur_mapping)
        """
        topo_name: str = choice(self.processor.topo_names)
        qubit_num = self.topo_qubit_num_map[topo_name]
        circ = Circuit(qubit_num)
        gate_num = randint(
            1, max(qubit_num * randint(3, self._max_layer_num) // 10, 30)
        )
        circ.random_append(
            gate_num,
            typelist=[
                GateType.cx,
            ],
        )
        layered_circ, success = self.processor.get_layered_circ(circ=circ)
        if not success:
            raise RuntimeError("Circuit layer exceed!!!")

        x = self.topo_x_map[topo_name]
        cur_mapping = [i for i in range(self.topo_qubit_num_map[topo_name])]
        circ_graph = self.processor.get_circ_graph(
            layered_circ=layered_circ,
            cur_mapping=cur_mapping,
            topo_dist=self.topo_dist_map[topo_name],
        )
        spacial_encoding = self.processor.get_spacial_encoding(
            circ_graph=circ_graph,
        )
        return layered_circ, topo_name, x, spacial_encoding, cur_mapping
