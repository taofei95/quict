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
        working_dir: str = None,
    ) -> None:
        if working_dir is None:
            working_dir = osp.dirname(osp.abspath(__file__))

        self._data_dir = osp.join(working_dir, "data")
        self._circ_dir = osp.join(self._data_dir, "circ")
        self._topo_dir = osp.join(self._data_dir, "topo")
        self._processed_dir = osp.join(self._data_dir, "processed_vnode")

        self._max_qubit_num = max_qubit_num

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
        for i in range(self._max_qubit_num + 1):
            g.add_node(i)
        for edge in topo.edge_list:
            edge: LayoutEdge
            g.add_edge(edge.u, edge.v)
        return g

    def _build_circ_repr(self, circ: Circuit, max_layer_num: int) -> nx.Graph:
        """Build a graph representation of a circuit.
        The circuit is divided into layers and each layer
        is built into a sub graph with respect to qubits.
        All subgraphs will be connected by qubits.

        Args:
            circ (Circuit): Circuit to be built.
            max_layer_num (int): Maximal number of layers of circuit. If the
            circuit has fewer layers, some empty layers will be padded. If the
            circuit has more layers, extra layers will be dropped.

        Returns:
            nx.Graph: Graph representation.
        """
        layers_raw: List[List[Tuple[int, int]]] = [[] for _ in range(max_layer_num)]
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
        for idx, layer in enumerate(layers_raw):
            offset = idx * self._max_qubit_num + 1
            for b in range(offset, offset + self._max_qubit_num):
                g.add_edge(b, 0)
            for u, v in layer:
                g.add_edge(u + offset, v + offset)
            # if idx > 0:
            #     for i in range(offset, offset + self._max_qubit_num):
            #         g.add_edge(i, i - self._max_qubit_num)
        return g

    def get_spacial_encoding(
        self, graph: nx.Graph, max_topology_diameter: int
    ) -> torch.IntTensor:
        """Build the spacial encoding of a given graph. A spacial
        encoding is similar to a shortest path matrix except that the
        shortest path corresponding with special virtual node is set as -1.

        Args:
            graph (nx.Graph): Graph to be handled. The input graph must
            have consecutive node indices starting from 0. You must
            guarantee that 0 is the virtual node.

        Returns:
            torch.IntTensor: Spacial encoding matrix WITHOUT embedding.
        """
        num_node = len(graph.nodes)
        _inf = max_topology_diameter + 2
        dist = [[_inf for _ in range(num_node)] for _ in range(num_node)]
        dist = torch.IntTensor(dist)
        for i in range(num_node):
            dist[i][i] = 0

        sp = nx.all_pairs_shortest_path_length(graph)
        for u, row in sp:
            for v, d in row.items():
                dist[u][v] = d

        v = 0
        for i in range(num_node):
            dist[v][i] = max_topology_diameter + 1
            dist[i][v] = max_topology_diameter + 1
        dist[v][v] = 0

        return dist

    def get_spacing_encoding_scale_factor(
        self, graph: nx.Graph, max_qubit_num: int, penalty_factor: float = 0.8
    ) -> torch.Tensor:
        """Get the scale factor for spacial encoding. This is
        used as penalty for remote layers in circuit.

        Args:
            graph (nx.Graph): Layered graph of a circuit, including a virtual node labeled with 0.
            max_qubit_num (int): Qubit number AFTER padding.

        Return:
            torch.Tensor: Factor tensor shaped as (n, 1).
        """
        num_node = len(graph.nodes)
        layer_num = (num_node - 1) // max_qubit_num
        assert layer_num * max_qubit_num + 1 == num_node

        penalty = 1
        factors = torch.empty((num_node, 1), dtype=torch.float)
        factors[0][0] = 1.0
        for layer in range(layer_num):
            for i in range(layer * max_qubit_num + 1, (layer + 1) * max_qubit_num + 1):
                factors[i][0] = penalty
            penalty *= penalty_factor
        return factors
