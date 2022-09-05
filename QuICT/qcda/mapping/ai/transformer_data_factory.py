import os
import networkx as nx
from random import choice, randint
import os.path as osp
from typing import Dict, Iterable, List, Tuple, Set
import torch
from QuICT.core import *
from QuICT.core.gate import GateType, BasicGate
from QuICT.core.layout import LayoutEdge
from QuICT.core.utils import CircuitBased
from numba import jit, njit
import numpy as np


@njit
def _floyd(n: int, dist: np.ndarray, inf: int) -> np.ndarray:
    for i in range(n):
        dist[i][i] = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    # Reset unconnected components with a special label.
    for i in range(n):
        for j in range(n):
            if dist[i][j] >= inf:
                dist[i][j] = 0
    return dist


@njit
def _fill_x(
    x: np.ndarray,
    max_qubit_num: int,
    max_layer_num: int,
    topo_qubit_num: int,
) -> np.ndarray:
    assert topo_qubit_num <= max_qubit_num

    # x = np.zeros(node_num, dtype=int)
    x[0] = max_qubit_num + 1
    for layer_idx in range(max_layer_num):
        offset = 1 + max_qubit_num * layer_idx
        for b in range(topo_qubit_num):
            x[b + offset] = b + 1
    return x


class CircuitTransformerDataFactory:
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        data_dir: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
        self._topo_dir = osp.join(data_dir, "topo")

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._node_num = max_qubit_num * (max_layer_num + 1)
        self._device = device

        self._circ_path_list = None
        self._topo_names = None

        # Topo attr cache def
        # These attributes maps are lazily initialized for faster start up.

        self._topo_graph_map = {}
        self._topo_edge_map = {}
        self._topo_qubit_num_map = {}
        self._topo_x_map = {}
        self._topo_mask_map = {}
        self._topo_dist_map = {}

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

    def _load_circuits_path(self) -> Iterable[Tuple[str, str]]:
        for topo_name in self.topo_names:
            for root, _, filenames in os.walk(osp.join(self._circ_dir, topo_name)):
                for name in filenames:
                    if name.startswith("result") or name.startswith("skip"):
                        continue
                    circ_path = osp.join(root, name)
                    yield topo_name, circ_path

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

    def get_layered_circ(
        self, circ: CircuitBased
    ) -> Tuple[List[Set[Tuple[int, int]]], bool]:
        """Get all 2-bit gate pairs by layers.

        Args:
            circ (CircuitBased): Input circuit.

        Returns:
            Tuple[List[Set[Tuple[int, int]]], bool]:
                List of layers, and a success flag. A layer is a set of pairs.
        """
        layers_raw: List[Set[Tuple[int, int]]] = []
        occupied = [-1 for _ in range(self._max_qubit_num)]
        for gate in circ.gates:
            gate: BasicGate
            if gate.controls + gate.targets != 2:
                continue
            a, b = gate.cargs + gate.targs
            idx = max(occupied[a], occupied[b]) + 1
            if idx >= self._max_layer_num:
                return None, False
            while idx >= len(layers_raw):
                layers_raw.append(set())
            layers_raw[idx].add((a, b))
            layers_raw[idx].add((b, a))
            occupied[a] = idx
            occupied[b] = idx
        return layers_raw, True

    def get_circ_edges(
        self,
        layered_circ: List[Set[Tuple[int, int]]],
        topo_dist: np.ndarray,
        cur_mapping: List[int] = None,
    ) -> Tuple[Tuple[int, int, int]]:
        """Build a graph representation of a circuit.
        The circuit is divided into layers and each layer
        is built into a sub graph with respect to qubits.
        All subgraphs will be connected by qubits.
        A virtual layer would be inserted and connected to the first layer
        by qubit.

        Args:
            layered_circ (List[Set[Tuple[int, int]]]): Circuit to be built. Circuit is
                represented in the layered 2-bit gate pair form.
                If circuit has fewer layers, some empty layers will be padded. If the
                circuit has more layers, extra layers will be dropped.
            topo_dist (np.ndarray): Physical qubit distance matrix.
            cur_mapping (List[int]): Current physical-logical qubit mapping.
                logical qubit u is mapped to physical qubit cur_mapping[v].
                If cur_mapping is not provided, the identical mapping will be used.

        Returns:
            Tuple[Tuple[int,int,int]]: All edges in circ graph.
        """
        if cur_mapping is None:
            cur_mapping = [i for i in range(self._max_qubit_num)]

        edges = []

        for layer_idx, layer in enumerate(layered_circ):
            offset = (layer_idx + 1) * self._max_qubit_num
            for b in range(self._max_qubit_num):
                prev = b + offset - self._max_qubit_num
                edges.append((prev, b + offset, 1))
            for u, v in layer:
                _u = cur_mapping[u]
                _v = cur_mapping[v]
                edges.append((_u + offset, _v + offset, topo_dist[_u][_v]))
                edges.append((_v + offset, _u + offset, topo_dist[_u][_v]))
        return edges

    def get_topo_dist(self, topo_graph: nx.Graph) -> np.ndarray:
        n = self._max_qubit_num
        inf = n + 100
        topo_dist = np.empty(shape=(n, n), dtype=int)
        topo_dist[:, :] = inf
        for u, v in topo_graph.edges:
            topo_dist[u][v] = 1
            topo_dist[v][u] = 1
        topo_dist = _floyd(n, topo_dist, inf)
        return topo_dist

    def get_spacial_encoding(
        self, circ_edges: Tuple[Tuple[int, int, int]]
    ) -> torch.IntTensor:
        """Build the spacial encoding of a given graph. The spacial encoding
        will be masked by corresponding physical topology. A spacial
        encoding is similar to a shortest path matrix except that the
        vertex pair we do not need is set as 0 distance.

        Args:
            circ_edges (Tuple[Tuple[int, int, int]]): All edges of circuit graph. The input graph must
                be remapped by current mapping.

        Returns:
            torch.IntTensor: Spacial encoding matrix WITHOUT embedding.
                If there's no available path between two nodes, the
                distance will be marked as 0.
        """
        n = self._node_num
        inf = n + 100
        dist = np.empty(shape=(n, n), dtype=int)
        dist[:, :] = inf
        for u, v, w in circ_edges:
            dist[u][v] = w
        dist = _floyd(n, dist, inf)
        dist = torch.from_numpy(dist).to(self._device)
        return dist

    def get_x(self, topo_qubit_num: int) -> torch.IntTensor:
        assert topo_qubit_num <= self._max_qubit_num

        x = np.zeros(self._node_num, dtype=int)
        x = _fill_x(
            x,
            max_qubit_num=self._max_qubit_num,
            max_layer_num=self._max_layer_num,
            topo_qubit_num=topo_qubit_num,
        )
        x = torch.from_numpy(x).to(self._device)
        return x

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
    def topo_dist_map(self) -> Dict[str, np.ndarray]:
        if len(self._topo_dist_map) == 0:
            self._reset_topo_attr_cache()
        return self._topo_dist_map

    def _reset_topo_attr_cache(self):
        for topo_name in self.topo_names:
            topo_path = osp.join(self._topo_dir, f"{topo_name}.layout")
            topo = Layout.load_file(topo_path)
            topo_graph = self.get_topo_graph(topo)
            self._topo_graph_map[topo_name] = topo_graph
            self._topo_qubit_num_map[topo_name] = topo.qubit_number
            self._topo_x_map[topo_name] = self.get_x(topo.qubit_number)
            self._topo_dist_map[topo_name] = self.get_topo_dist(topo_graph=topo_graph)

            topo_mask = torch.zeros(
                (topo.qubit_number, topo.qubit_number),
                dtype=torch.float,
                device=self._device,
            )
            topo_edge = []
            for u, v in topo_graph.edges:
                topo_mask[u][v] = 1.0
                topo_mask[v][u] = 1.0
                topo_edge.append((u, v))
                topo_edge.append((v, u))
            self._topo_mask_map[topo_name] = topo_mask
            self._topo_edge_map[topo_name] = topo_edge

    @classmethod
    def remap_layered_circ(
        cls, layered_circ: List[Set[Tuple[int, int]]], cur_mapping: List[int]
    ) -> List[Set[Tuple[int, int]]]:
        new_layers = [set() for _ in range(len(layered_circ))]
        for layer_idx, layer in enumerate(layered_circ):
            for u, v in layer:
                _u = cur_mapping[u]
                _v = cur_mapping[v]
                new_layers[layer_idx].add((_u, _v))
                new_layers[layer_idx].add((_v, _u))
        return new_layers

    def get_one(
        self,
    ) -> Tuple[
        List[Set[Tuple[int, int]]], str, torch.IntTensor, torch.IntTensor, List[int]
    ]:
        """
        Returns:
            Tuple[List[Set[Tuple[int, int]]], str, torch.IntTensor, torch.IntTensor]: (layered_circ, topo_name, x, spacial_encoding, cur_mapping)
        """
        topo_name: str = choice(self.topo_names)
        qubit_num = self.topo_qubit_num_map[topo_name]
        circ = Circuit(qubit_num)

        success = False
        while not success:
            circ.gates.clear()
            gate_num = randint(
                1, max(qubit_num * randint(3, self._max_layer_num) // 5, 30)
            )
            circ.random_append(
                gate_num,
                typelist=[
                    GateType.cx,
                ],
            )
            layered_circ, success = self.get_layered_circ(circ=circ)

        x = self.topo_x_map[topo_name]
        cur_mapping = [i for i in range(self.topo_qubit_num_map[topo_name])]
        circ_edges = self.get_circ_edges(
            layered_circ=layered_circ,
            cur_mapping=cur_mapping,
            topo_dist=self.topo_dist_map[topo_name],
        )
        spacial_encoding = self.get_spacial_encoding(
            circ_edges=circ_edges,
        )
        return layered_circ, topo_name, x, spacial_encoding, cur_mapping
