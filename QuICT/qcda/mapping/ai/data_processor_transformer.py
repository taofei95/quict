import os
import os.path as osp
from typing import Iterable, List, Set, Tuple

import networkx as nx
import torch
from QuICT.core import *
from QuICT.core.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from QuICT.core.utils.circuit_info import CircuitBased
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


class CircuitTransformerDataProcessor:
    def __init__(
        self,
        max_qubit_num: int,
        max_layer_num: int,
        data_dir: str = None,
    ) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.abspath(__file__))
            data_dir = osp.join(data_dir, "data")

        self._data_dir = data_dir
        self._circ_dir = osp.join(self._data_dir, "circ")
        self._topo_dir = osp.join(self._data_dir, "topo")
        self._processed_dir = osp.join(self._data_dir, "processed_graphormer")

        self._max_qubit_num = max_qubit_num
        self._max_layer_num = max_layer_num
        self._node_num = max_qubit_num * (max_layer_num + 1)

        self._circ_path_list = None

        self._topo_names = None

    @property
    def circ_path_list(self) -> List[Tuple[str, str]]:
        if self._circ_path_list is None:
            self._circ_path_list = list(self._load_circuits_path())
        return self._circ_path_list

    def get_meta(self) -> Tuple[int, int]:
        """Get metadata of produced dataset.

        Returns:
            Tuple[int, int]: (max_qubit_num, max_layer_num).
        """
        return self._max_qubit_num, self._max_layer_num

    def _show_layer_distribution(self):
        distribution = {}
        for idx, (_, circ_path) in enumerate(self.circ_path_list):
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} files.")

            circ = OPENQASMInterface.load_file(circ_path).circuit
            layers_raw: List[List[Tuple[int, int]]] = []
            occupied = [-1 for _ in range(self._max_qubit_num)]
            for gate in circ.gates:
                gate: BasicGate
                if gate.controls + gate.targets != 2:
                    continue
                a, b = gate.cargs + gate.targs
                idx = max(occupied[a], occupied[b]) + 1
                if idx >= len(layers_raw):
                    layers_raw.append([])
                layers_raw[idx].append((a, b))
                layers_raw[idx].append((b, a))
                occupied[a] = idx
                occupied[b] = idx

            l = len(layers_raw)
            if l not in distribution:
                distribution[l] = 0
            distribution[l] += 1
        print(distribution)

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
        layers_raw: List[Set[Tuple[int, int]]] = [
            set() for _ in range(self._max_layer_num)
        ]
        occupied = [-1 for _ in range(self._max_qubit_num)]
        for gate in circ.gates:
            gate: BasicGate
            if gate.controls + gate.targets != 2:
                continue
            a, b = gate.cargs + gate.targs
            idx = max(occupied[a], occupied[b]) + 1
            if idx >= self._max_layer_num:
                return None, False
            layers_raw[idx].add((a, b))
            layers_raw[idx].add((b, a))
            occupied[a] = idx
            occupied[b] = idx
        return layers_raw, True

    # def remap_layered_circ(
    #     self, layered_circ: List[Set[Tuple[int, int]]], cur_mapping: List[int]
    # ) -> List[Set[Tuple[int, int]]]:
    #     new_layers = [set() for _ in range(self._max_layer_num)]
    #     for layer_idx, layer in enumerate(layered_circ):
    #         for u, v in layer:
    #             _u = cur_mapping[u]
    #             _v = cur_mapping[v]
    #             new_layers[layer_idx].add((_u, _v))
    #             new_layers[layer_idx].add((_v, _u))
    #     return new_layers

    def get_circ_graph(
        self,
        layered_circ: List[Set[Tuple[int, int]]],
        topo_dist: torch.IntTensor,
        cur_mapping: List[int] = None,
    ) -> nx.Graph:
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
            topo_dist (torch.IntTensor): Physical qubit distance matrix.
            cur_mapping (List[int]): Current physical-logical qubit mapping.
                logical qubit u is mapped to physical qubit cur_mapping[v].
                If cur_mapping is not provided, the identical mapping will be used.

        Returns:
            nx.Graph: Graph representation.
        """
        if cur_mapping is None:
            cur_mapping = [i for i in range(self._max_qubit_num)]

        while len(layered_circ) < self._max_layer_num:
            layered_circ.append([])

        g = nx.Graph()
        for i in range(self._node_num):
            g.add_node(i)
        for layer_idx, layer in enumerate(layered_circ):
            offset = (layer_idx + 1) * self._max_qubit_num
            for b in range(self._max_qubit_num):
                prev = b + offset - self._max_qubit_num
                g.add_edge(b + offset, prev, weight=1)
            for u, v in layer:
                _u = cur_mapping[u]
                _v = cur_mapping[v]
                g.add_edge(_u + offset, _v + offset, weight=topo_dist[_u][_v])
        return g

    def get_topo_dist(self, topo_graph: nx.Graph) -> torch.IntTensor:
        topo_dist = torch.zeros(
            (self._max_qubit_num, self._max_qubit_num), dtype=torch.int
        )
        sp = nx.all_pairs_shortest_path_length(topo_graph)
        for u, row in sp:
            for v, d in row.items():
                topo_dist[u][v] = d
                topo_dist[v][u] = d
        return topo_dist

    def get_spacial_encoding(self, circ_graph: nx.Graph) -> torch.IntTensor:
        """Build the spacial encoding of a given graph. The spacial encoding
        will be masked by corresponding physical topology. A spacial
        encoding is similar to a shortest path matrix except that the
        vertex pair we do not need is set as 0 distance.

        Args:
            circ_graph (nx.Graph): Graph to be handled. The input graph must
                be remapped by current mapping.

        Returns:
            torch.IntTensor: Spacial encoding matrix WITHOUT embedding.
                If there's no available path between two nodes, the
                distance will be marked as 0.
        """
        dist = torch.zeros((self._node_num, self._node_num), dtype=torch.int)
        sp = nx.all_pairs_dijkstra_path_length(circ_graph)
        for u, row in sp:
            for v, d in row.items():
                dist[u][v] = d
                dist[v][u] = d
        return dist

    def get_x(self, topo_qubit_number: int) -> torch.Tensor:
        assert topo_qubit_number <= self._max_qubit_num

        x = [0 for _ in range(self._node_num)]
        x[0] = self._max_qubit_num + 1
        for layer_idx in range(self._max_layer_num):
            offset = 1 + self._max_qubit_num * layer_idx
            for b in range(topo_qubit_number):
                x[b + offset] = b + 1
        x = torch.tensor(x, dtype=torch.int)
        return x

    def _build(self) -> Iterable[Tuple[List[Set[Tuple[int, int]]], str]]:
        """Build all circuit representations.

        Returns:
            Iterable[Tuple[List[Set[Tuple[int, int]]], str]]: Generator of all layered circuits and its topology name.
        """
        for topo_name, circ_path in self.circ_path_list:
            circ: Circuit = OPENQASMInterface.load_file(circ_path).circuit
            layered_circ, success = self.get_layered_circ(circ)
            if not success:
                continue
            yield layered_circ, topo_name

    def build(self):
        """Build all circuit representations and save them into process_dir."""
        if not osp.exists(self._processed_dir):
            print("No directory to save data. Create one.")
            os.makedirs(self._processed_dir)

        print("Start building...")

        metadata = self.get_meta()
        m_path = osp.join(self._processed_dir, "metadata.pt")
        torch.save(metadata, m_path)

        for idx, data in enumerate(self._build()):
            if (idx + 1) % 100 == 0:
                progress = (idx + 1) * 100 / len(self.circ_path_list)
                print(f"Processed {idx+1} files. Current progress: {progress:0.2f}%.")
            path = osp.join(self._processed_dir, f"{idx}.pt")
            torch.save(data, path)


if __name__ == "__main__":
    processor = CircuitTransformerDataProcessor(max_qubit_num=30, max_layer_num=70)
    # processor._show_layer_distribution()
    processor.build()
