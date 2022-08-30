import os
import os.path as osp
from typing import Iterable, List, Tuple

import networkx as nx
import torch
from QuICT.core import *
from QuICT.core.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge


class CircuitVnodeProcessor:
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        working_dir: str = None,
    ) -> None:
        if working_dir is None:
            working_dir = osp.dirname(osp.abspath(__file__))

        self._data_dir = osp.join(working_dir, "data")
        self._circ_dir = osp.join(self._data_dir, "circ")
        self._topo_dir = osp.join(self._data_dir, "topo")
        self._processed_dir = osp.join(self._data_dir, "processed_graphormer")

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._max_dist = 2 * max_layer_num

        self._circ_path_list = list(self._load_circuits_path())

    def _get_topo_names(self) -> Iterable[str]:
        for _, _, filenames in os.walk(self._topo_dir):
            for name in filenames:
                yield name.split(".")[0]

    def _load_circuits_path(self) -> Iterable[Tuple[str, str, str]]:
        for topo_name in self._get_topo_names():
            topo_path = osp.join(self._topo_dir, f"{topo_name}.layout")
            for root, _, filenames in os.walk(osp.join(self._circ_dir, topo_name)):
                for name in filenames:
                    if name.startswith("result") or name.startswith("skip"):
                        continue
                    circ_path = osp.join(root, name)
                    result_circ_path = osp.join(root, f"result_{name}")
                    yield topo_path, circ_path, result_circ_path

    def _build_topo_repr(self, topo: Layout) -> nx.Graph:
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

    def _build_circ_repr(self, circ: Circuit) -> nx.Graph:
        """Build a graph representation of a circuit.
        The circuit is divided into layers and each layer
        is built into a sub graph with respect to qubits.
        All subgraphs will be connected by qubits.
        A virtual node (labeled as 0) will be connected with all the qubits in the first layer.

        Args:
            circ (Circuit): Circuit to be built.
                If circuit has fewer layers, some empty layers will be padded. If the
                circuit has more layers, extra layers will be dropped.

        Returns:
            nx.Graph: Graph representation.
        """
        layers_raw: List[List[Tuple[int, int]]] = [
            [] for _ in range(self._max_layer_num)
        ]
        occupied = [-1 for _ in range(self._max_qubit_num)]
        for gate in circ.gates:
            gate: BasicGate
            if gate.controls + gate.targets != 2:
                continue
            a, b = gate.cargs + gate.targs
            idx = max(occupied[a], occupied[b]) + 1
            if idx >= len(layers_raw):
                continue
            layers_raw[idx].append((a, b))
            layers_raw[idx].append((b, a))
            occupied[a] = idx
            occupied[b] = idx
        g = nx.Graph()
        for i in range(self._max_qubit_num * len(layers_raw) + 1):
            g.add_node(i)
        for layer_idx, layer in enumerate(layers_raw):
            offset = layer_idx * self._max_qubit_num + 1
            for b in range(self._max_qubit_num):
                prev = 0 if layer_idx == 0 else b + offset - self._max_qubit_num
                g.add_edge(b + offset, prev)
            for u, v in layer:
                g.add_edge(u + offset, v + offset)
        return g

    def get_spacial_encoding(
        self, circ_graph: nx.Graph, topo: Layout = None
    ) -> torch.IntTensor:
        """Build the spacial encoding of a given graph. The spacial encoding
        will be masked by corresponding physical topology. A spacial
        encoding is similar to a shortest path matrix except that the
        vertex pair we do not need is set as 0 distance.

        Args:
            circ_graph (nx.Graph): Graph to be handled. The input graph must
                have consecutive node indices starting from 0. You must
                guarantee that 0 is the virtual node.
            topo (Layout): Topology of the physical device.

        Returns:
            torch.IntTensor: Spacial encoding matrix WITHOUT embedding.
                If there's no available path between two nodes, the
                distance will be marked as 0.
        """
        num_node = len(circ_graph.nodes)
        dist = [[0 for _ in range(num_node)] for _ in range(num_node)]
        dist = torch.IntTensor(dist)

        topo_graph = self._build_topo_repr(topo=topo)
        topo_dist = [
            [0 for _ in range(self._max_qubit_num)] for _ in range(self._max_qubit_num)
        ]
        sp = nx.all_pairs_shortest_path_length(topo_graph)
        for u, row in sp:
            for v, d in row.items():
                topo_dist[u][v] = d
                topo_dist[v][u] = d
        topo_dist = torch.IntTensor(topo_dist)

        # Virtual node and first layer
        for b in range(self._max_qubit_num):
            dist[0][b + 1] = 1
            dist[b + 1][0] = 1
        for layer_idx in range(1, self._max_layer_num):
            offset = 1 + layer_idx * self._max_qubit_num
            # Inter-layer connections
            for b in range(self._max_qubit_num):
                prev = b + offset - self._max_qubit_num
                dist[b][prev] = 1
                dist[prev][b] = 1
            # Inner-layer connections
            for u in range(self._max_qubit_num):
                for v in range(self._max_qubit_num):
                    if circ_graph.has_edge(u + offset, v + offset):
                        dist[u + offset][v + offset] = topo_dist[u][v]
                        dist[v + offset][u + offset] = topo_dist[u][v]

        return dist


if __name__ == "__main__":
    pass
