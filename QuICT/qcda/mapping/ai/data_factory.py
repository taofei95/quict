import os
import os.path as osp
from random import choice, randint
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import torch
from numba import njit
from QuICT.core import *
from QuICT.core.gate import BasicGate, GateType, CompositeGate
from QuICT.core.layout import LayoutEdge
from QuICT.core.utils import CircuitBased
from __future__ import annotations


@njit
def _floyd(n: int, dist: np.ndarray, _inf: int) -> np.ndarray:
    for i in range(n):
        dist[i][i] = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    for i in range(n):
        for j in range(n):
            if dist[i][j] >= _inf:
                dist[i][j] = 0
    return dist


# class GateLayer(CompositeGate):
#     def __init__(self, max_qubit: int, idx: int) -> None:
#         super().__init__()
#         self.idx = idx
#         self._max_qubit = max_qubit
#         self.prev = [None for _ in range(max_qubit)]
#         self.next = [None for _ in range(max_qubit)]

#     def append(self, gate: BasicGate, prev: Dict[int, GateLayer]):
#         super().append(gate)
#         for idx, layer in prev.items():
#             self.prev[idx] = layer
#             layer.next[idx] = self


class DataFactory:
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        data_dir: str = None,
    ) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
        self._data_dir = data_dir
        self._topo_dir = osp.join(data_dir, "topo")

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._node_num = max_qubit_num * (max_layer_num + 1)

        self._circ_path_list = None
        self._topo_names = None

        # Topo attr cache def
        # These attributes maps are lazily initialized for faster start up.

        self._topo_map = {}
        self._topo_graph_map = {}
        self._topo_edge_map = {}
        self._topo_edge_mat_map = {}
        self._topo_qubit_num_map = {}
        self._topo_mask_map = {}
        self._topo_dist_map = {}
        self._x: torch.LongTensor = None

    @property
    def _x_raw(self) -> torch.LongTensor:
        if self._x is None:
            x = 1 + torch.arange(self._node_num, dtype=torch.long)
            # x = torch.tensor(x, dtype=torch.long)
            self._x = x
        return self._x

    @property
    def circ_path_list(self) -> List[Tuple[str, str]]:
        if self._circ_path_list is None:
            self._circ_path_list = list(self._load_circuits_path())
        return self._circ_path_list

    @property
    def topo_names(self) -> List[str]:
        if self._topo_names is None:
            self._topo_names = []
            for _, _, filenames in os.walk(self._topo_dir):
                for name in filenames:
                    self._topo_names.append(name.split(".")[0])
        return self._topo_names

    @property
    def topo_graph_map(self) -> Dict[str, nx.Graph]:
        if len(self._topo_graph_map) == 0:
            self._reset_attr_cache()
        return self._topo_graph_map

    @property
    def topo_edge_map(self) -> Dict[str, List[Tuple[int, int]]]:
        if len(self._topo_edge_map) == 0:
            self._reset_attr_cache()
        return self._topo_edge_map

    @property
    def topo_qubit_num_map(self) -> Dict[str, int]:
        if len(self._topo_qubit_num_map) == 0:
            self._reset_attr_cache()
        return self._topo_qubit_num_map

    @property
    def topo_mask_map(self) -> Dict[str, torch.Tensor]:
        if len(self._topo_mask_map) == 0:
            self._reset_attr_cache()
        return self._topo_mask_map

    @property
    def topo_dist_map(self) -> Dict[str, np.ndarray]:
        if len(self._topo_dist_map) == 0:
            self._reset_attr_cache()
        return self._topo_dist_map

    @property
    def topo_edge_mat_map(self) -> Dict[str, np.ndarray]:
        if len(self._topo_edge_mat_map) == 0:
            self._reset_attr_cache()
        return self._topo_edge_mat_map

    @property
    def topo_map(self) -> Dict[str, Layout]:
        if len(self._topo_map) == 0:
            self._reset_attr_cache()
        return self._topo_map

    def _reset_attr_cache(self):
        for topo_name in self.topo_names:
            topo_path = osp.join(self._topo_dir, f"{topo_name}.layout")
            topo = Layout.load_file(topo_path)
            self._topo_map[topo_name] = topo
            topo_graph = self.get_topo_graph(topo)
            self._topo_graph_map[topo_name] = topo_graph
            self._topo_qubit_num_map[topo_name] = topo.qubit_number
            self._topo_dist_map[topo_name] = self.get_topo_dist(topo_graph=topo_graph)

            topo_mask = torch.zeros(
                (self._max_qubit_num, self._max_qubit_num), dtype=torch.float
            )

            topo_edge = []
            topo_adj_mat_thin = np.zeros(
                (topo.qubit_number, topo.qubit_number), dtype=int
            )
            for u, v in topo_graph.edges:
                topo_mask[u][v] = 1.0
                topo_mask[v][u] = 1.0
                topo_edge.append((u, v))
                topo_edge.append((v, u))
                topo_adj_mat_thin[u][v] = 1
                topo_adj_mat_thin[v][u] = 1
            self._topo_mask_map[topo_name] = topo_mask
            self._topo_edge_map[topo_name] = topo_edge
            self._topo_edge_mat_map[topo_name] = topo_adj_mat_thin

    def get_topo_graph(self, topo: Layout) -> nx.Graph:
        """Build tha graph representation of a topology.
        Then add a virtual node (labeled 0) into it.

        Args:
            topo (Layout): Topology to be built.

        Returns:
            nx.Graph: Graph representation.
        """
        g = nx.Graph()
        for i in range(self._max_qubit_num):
            g.add_node(i)
        for edge in topo.edge_list:
            edge: LayoutEdge
            g.add_edge(edge.u, edge.v)
        return g

    def get_topo_dist(self, topo_graph: nx.Graph) -> np.ndarray:
        _inf = self._node_num + 5
        n = self._max_qubit_num
        dist = np.empty((n, n), dtype=np.int)
        dist[:, :] = _inf
        for u, v in topo_graph.edges:
            dist[u][v] = 1
            dist[v][u] = 1
        dist = _floyd(n, dist, _inf)
        return dist

    def get_topo_mask(self, topo_graph: nx.Graph) -> torch.Tensor:
        topo_mask = torch.zeros(
            (self._max_qubit_num, self._max_qubit_num), dtype=torch.float
        )
        for u, v in topo_graph.edges:
            topo_mask[u][v] = 1.0
            topo_mask[v][u] = 1.0
        return topo_mask

    def get_topo_edges(self, topo_graph: nx.Graph) -> np.ndarray:
        topo_edge = []
        for u, v in topo_graph.edges:
            topo_edge.append((u, v))
            topo_edge.append((v, u))
        return topo_edge

    def get_layered_circ(
        self, circ: CircuitBased
    ) -> Tuple[List[Dict[Tuple[int, int], BasicGate]], bool]:
        """Get all 2-bit gate pairs by layers.

        Args:
            circ (CircuitBased): Input circuit.

        Returns:
            Tuple[List[Dict[Tuple[int, int], BasicGate]], bool]:
                List of layers, and a success flag. A layer is a map from qubit-pair to quantum gate.
        """
        layers_raw: List[Dict[Tuple[int, int], BasicGate]] = []
        occupied = [-1 for _ in range(self._max_qubit_num)]
        flag = True
        for gate in circ.gates:
            gate: BasicGate
            if gate.controls + gate.targets != 2:
                continue
            a, b = gate.cargs + gate.targs
            idx = max(occupied[a], occupied[b]) + 1
            if idx >= self._max_layer_num:
                flag = False
            while idx >= len(layers_raw):
                layers_raw.append(set())
            layers_raw[idx][(a, b)] = gate
            layers_raw[idx][(b, a)] = gate
            occupied[a] = idx
            occupied[b] = idx
        return layers_raw, flag

    def get_x(self) -> torch.LongTensor:
        return self._x_raw

    def get_circ_edge_index(
        self,
        layered_circ: List[Dict[Tuple[int, int], BasicGate]],
        topo_graph: nx.Graph,
        logic2phy: List[int],
    ) -> torch.IntTensor:
        """Build the edge index of circuit graph. A virtual layer will be appended first
        to gather information and fuse topology.

        Args:
            layered_circ (List[Dict[Tuple[int, int], BasicGate]]): Layered circuit representation after remapped by current mapping.
            topo_graph (nx.Graph): Topology graph.
            logic2phy (List[int]): Current logical to physical mapping.

        Returns:
            torch.IntTensor
        """
        q = self._max_qubit_num
        edge_index = []

        # Virtual layer is used for topology.
        for u, v in topo_graph.edges:
            edge_index.append((u, v))
            edge_index.append((v, u))

        for layer_idx, layer in enumerate(layered_circ):
            offset = q * (layer_idx + 1)
            for b in range(q):
                _b = b + offset
                # Directed edge from previous layer to current layer?
                edge_index.append((_b - q, _b))
                edge_index.append((_b, _b - q))
            for u, v in layer.keys():
                _u = logic2phy[u] + offset
                _v = logic2phy[v] + offset
                edge_index.append((_u, _v))
                edge_index.append((_v, _u))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return edge_index

    # @classmethod
    # def remap_layered_circ(
    #     cls, layered_circ: List[Set[Tuple[int, int]]], cur_mapping: List[int]
    # ) -> List[Set[Tuple[int, int]]]:
    #     new_layers = [set() for _ in range(len(layered_circ))]
    #     for layer_idx, layer in enumerate(layered_circ):
    #         for u, v in layer:
    #             _u = cur_mapping[u]
    #             _v = cur_mapping[v]
    #             new_layers[layer_idx].add((_u, _v))
    #             # new_layers[layer_idx].add((_v, _u))
    #     return new_layers

    def get_one(
        self, topo_name: str = None
    ) -> Tuple[
        List[Set[Tuple[int, int]]],
        Layout,
        torch.Tensor,
        nx.Graph,
        np.ndarray,
        torch.LongTensor,
        torch.LongTensor,
        List[int],
    ]:
        if topo_name is None:
            topo_name: str = choice(self.topo_names)
        # topo_name: str = "ibmq_peekskill"
        topo = self.topo_map[topo_name]
        qubit_num = self.topo_qubit_num_map[topo_name]
        topo_graph = self.topo_graph_map[topo_name]
        topo_dist = self.get_topo_dist(topo_graph=topo_graph)
        topo_edges = tuple(self.topo_edge_map[topo_name])
        circ = Circuit(qubit_num)

        success = False
        circ.gates.clear()
        min_gn = 80
        gate_num = randint(
            min_gn, max(qubit_num // 2 * randint(2, self._max_layer_num), min_gn)
        )
        circ.random_append(
            gate_num,
            typelist=[
                GateType.cx,
            ],
        )
        layered_circ, success = self.get_layered_circ(circ=circ)
        while len(layered_circ) > self._max_layer_num:
            layered_circ.pop(-1)

        x = self.get_x()
        cur_mapping = [i for i in range(self.topo_qubit_num_map[topo_name])]
        edge_index = self.get_circ_edge_index(
            layered_circ=layered_circ,
            topo_graph=topo_graph,
            logic2phy=cur_mapping,
        )
        return (
            layered_circ,
            topo,
            self.topo_mask_map[topo_name],
            topo_graph,
            topo_dist,
            topo_edges,
            x,
            edge_index,
            cur_mapping,
        )
