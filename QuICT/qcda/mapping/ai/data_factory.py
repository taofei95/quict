import copy
import math
import os
import os.path as osp
from random import choice, randint
import random
from typing import Dict, Iterator, List, Set, Tuple, Union

import networkx as nx
import numpy as np
import torch
from numba import njit
from QuICT.core import *
from QuICT.core.gate import BasicGate, CompositeGate, GateType
from QuICT.core.layout import LayoutEdge
from QuICT.core.utils.circuit_info import CircuitBased
from torch_geometric.data import Data as PygData
from torch_geometric.utils import from_networkx


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


class CircuitState:
    """DAG Representation of a quantum circuit. A virtual node will be
    added with label 0.
    """

    def __init__(
        self, circ: Union[Circuit, CompositeGate, List[BasicGate]], max_gate_num: int
    ) -> None:
        self._qubit_num = circ.width()
        self._max_gate_num = max_gate_num
        q = circ.width()
        if isinstance(circ, CircuitBased):
            self._gates: List[BasicGate] = copy.deepcopy(circ.gates)
        elif isinstance(circ, list):
            self._gates = copy.deepcopy(circ)
        else:
            raise TypeError(
                "circ argument only supports Circuit/CompositeGate/List[BasicGate]"
            )

        self._graph = nx.DiGraph()
        # self._graph.add_node(0)
        for gid in range(len(self._gates)):
            # self._graph.add_node(gid + 1)
            self._graph.add_node(gid)

        self._occupied = [-1 for _ in range(q)]
        self._bit2gid: List[List[int]] = [[] for _ in range(q)]
        """Qubit to all gates on it.
        """
        for gid, gate in enumerate(self._gates):
            self.add_gate(gid, gate, 1)

        self._removed_cnt = 0

    def add_gate(self, gid, gate: BasicGate, tag: int):
        assert gate.controls + gate.targets == 2, "Only 2 bit gates are supported."
        a, b = gate.cargs + gate.targs
        assert a != b
        # Position to Gate ID
        self._bit2gid[a].append(gid)
        self._bit2gid[b].append(gid)
        # DAG edges
        if self._occupied[a] != -1:
            self._graph.add_edge(self._occupied[a], gid)
            # self._graph.add_edge(gid, self._occupied[a])
        if self._occupied[b] != -1:
            self._graph.add_edge(self._occupied[b], gid)
            # self._graph.add_edge(gid, self._occupied[b])
        nx.set_node_attributes(self._graph, {gid: {"gid": gid}})
        self._occupied[a] = gid
        self._occupied[b] = gid

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def copy(self):
        return copy.deepcopy(self)

    def count_gate(self) -> int:
        return nx.number_of_nodes(self._graph)

    def first_layer_gates(self) -> Dict[int, BasicGate]:
        ans = {}
        for bit_stick in self._bit2gid:
            if not bit_stick:
                # Skip if empty
                continue
            gid = bit_stick[0]
            gate = self._gates[gid]
            a, b = gate.cargs + gate.targs
            if self._bit2gid[a][0] == self._bit2gid[b][0] and gid not in ans:
                ans[gid] = self._gates[gid]
        return ans

    def eager_exec(
        self,
        logic2phy: List[int],
        topo_graph: nx.DiGraph,
        physical_circ: CompositeGate = None,
    ) -> int:
        """Eagerly remove all executable gates for now.

        Args:
            logic2phy (List[int]): Current logical to physical qubit mapping.
            topo_graph (nx.DiGraph): Physical topology graph.
            physical_circ (CompositeGate, optional): If set, executed gates are appended to it.

        Returns:
            int: Removed gate number.
        """
        remove_cnt = 0
        remove_any = True
        while remove_any:
            remove_any = False
            for gid, gate in self.first_layer_gates().items():
                a, b = gate.cargs + gate.targs
                _a, _b = logic2phy[a], logic2phy[b]
                if topo_graph.has_edge(_a, _b):
                    self._bit2gid[a].pop(0)
                    self._bit2gid[b].pop(0)
                    remove_cnt += 1
                    remove_any = True
                    self._graph.remove_node(gid)
                    if physical_circ is not None:
                        with physical_circ:
                            gate & [_a, _b]
        self._removed_cnt += remove_cnt
        return remove_cnt

    def remained_circ(self, logic2phy: List[int]) -> CompositeGate:
        gates = {}
        for bit_stick in self._bit2gid:
            for gid in bit_stick:
                if gid not in gates:
                    gates[gid] = self._gates[gid]
        cg = CompositeGate()
        for gate in gates.values():
            a, b = gate.cargs + gate.targs
            a, b = logic2phy[a], logic2phy[b]
            with cg:
                gate & [a, b]
        return cg

    def biased_random_swap(
        self, topo_dist: np.ndarray, logic2phy: List[int], zero_shift: float = 0.1
    ) -> Tuple[int, int]:
        candidates = []
        weights = []

        for i in range(self._qubit_num):
            for j in range(i + 1, self._qubit_num):
                if abs(topo_dist[i][j] - 1) > 1e-6:
                    continue
                candidates.append((i, j))
                next_logic2phy = copy.copy(logic2phy)
                next_logic2phy[i], next_logic2phy[j] = (
                    next_logic2phy[j],
                    next_logic2phy[i],
                )
                bias = 0
                for gate in self.first_layer_gates().values():
                    a, b = gate.cargs + gate.targs
                    _a, _b = logic2phy[a], logic2phy[b]
                    prev_d = topo_dist[_a][_b]
                    _a, _b = next_logic2phy[a], next_logic2phy[b]
                    next_d = topo_dist[_a][_b]
                    bias += prev_d - next_d
                if abs(bias) < 1e-6:
                    bias = zero_shift
                bias = max(0, bias)
                weights.append(bias)
        assert len(candidates) > 0
        if random.random() < 0.5:
            action = random.choices(population=candidates, weights=weights, k=1)[0]
        else:
            action = random.choices(population=candidates, k=1)[0]
        return action

    def to_pyg(self, logic2phy: List[int]) -> PygData:
        """Convert current data into PyG Data according to current mapping.

        Arg:
            logic2phy (List[int]): Logical to physical qubit mapping.
            max_qubit_num (int): Padding size.

        Returns:
            PygData: PyG data.
        """
        x = torch.zeros(self._max_gate_num, 2, dtype=torch.long)
        edge_index = []
        # g = nx.convert_node_labels_to_integers(self._graph)
        g = self._graph
        for node in g.nodes:
            gid = g.nodes[node]["gid"]
            gate = self._gates[gid]
            a, b = gate.cargs + gate.targs
            x[node][0] = logic2phy[a] + 1
            x[node][1] = logic2phy[b] + 1
            # Add self loops
            edge_index.append([gid, gid])
        for u, v in self._graph.edges:
            edge_index.append([u, v])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
        data = PygData(x=x, edge_index=edge_index)
        return data


class State:
    def __init__(
        self,
        circ_graph: CircuitState,
        topo: Layout,
        topo_mask: torch.Tensor,
        topo_graph: nx.DiGraph,
        topo_dist: np.ndarray,
        topo_edges: Tuple[Tuple[int, int]],
        circ_pyg_data: PygData,
        logic2phy: List[int],
        # phy2logic: List[int] = None,
    ) -> None:
        self.circ_state = circ_graph
        self.topo = topo
        self.topo_mask = topo_mask
        self.topo_graph = topo_graph
        self.topo_dist = topo_dist
        self.topo_edges = topo_edges
        self.circ_pyg_data = circ_pyg_data
        self.logic2phy = logic2phy
        """Logical to physical mapping
        """
        # self.phy2logic = phy2logic
        # """Physical to logical mapping
        # """
        # if phy2logic is None:
        #     self.phy2logic = [-1 for _ in range(len(logic2phy))]
        #     for i in range(len(logic2phy)):
        #         self.phy2logic[logic2phy[i]] = i

    def remained_circ(self) -> CompositeGate:
        return self.circ_state.remained_circ(self.logic2phy)

    def biased_random_swap(self) -> Tuple[int, int]:
        return self.circ_state.biased_random_swap(self.topo_dist, self.logic2phy)


class Transition:
    def __init__(
        self, state: State, action: Tuple[int, int], next_state: State, reward: float
    ) -> None:
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __iter__(self):
        return iter((self.state, self.action, self.next_state, self.reward))


class DataFactory:
    def __init__(
        self,
        topo: Union[str, Layout],
        max_gate_num: int,
        data_dir: str = None,
    ) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
        self._data_dir = data_dir
        self._topo_dir = osp.join(data_dir, "topo")

        self._max_gate_num = max_gate_num

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

        if isinstance(topo, str):
            self._cur_topo = self.topo_map[topo]
        elif isinstance(topo, Layout):
            self._cur_topo = topo
        else:
            raise TypeError("Only support layout name or Layout object.")
        assert self._cur_topo is not None

    @property
    def topo_names(self) -> List[str]:
        if self._topo_names is None:
            self._topo_names = []
            for _, _, filenames in os.walk(self._topo_dir):
                for name in filenames:
                    self._topo_names.append(name.split(".")[0])
        return self._topo_names

    @property
    def topo_graph_map(self) -> Dict[str, nx.DiGraph]:
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
            topo_path = osp.join(self._topo_dir, f"{topo_name}.json")
            topo = Layout.load_file(topo_path)
            self._topo_map[topo_name] = topo
            topo_graph = self._get_topo_graph(topo)
            self._topo_graph_map[topo_name] = topo_graph
            self._topo_qubit_num_map[topo_name] = topo.qubit_number
            self._topo_dist_map[topo_name] = self._get_topo_dist(topo_graph=topo_graph)

            topo_mask = self._get_topo_mask(topo_graph=topo_graph)

            topo_edge = self._get_topo_edges(topo_graph=topo_graph)
            topo_adj_mat_thin = np.zeros(
                (topo.qubit_number, topo.qubit_number), dtype=int
            )
            for u, v in topo_graph.edges:
                topo_adj_mat_thin[u][v] = 1
            self._topo_mask_map[topo_name] = topo_mask
            self._topo_edge_map[topo_name] = topo_edge
            self._topo_edge_mat_map[topo_name] = topo_adj_mat_thin

    def _get_topo_graph(self, topo: Layout) -> nx.DiGraph:
        """Build tha graph representation of a topology.

        Args:
            topo (Layout): Topology to be built.

        Returns:
            nx.DiGraph: Graph representation.
        """
        g = nx.DiGraph()
        # g.add_node(0)
        for i in range(topo.qubit_number):
            g.add_node(i)
            # g.add_node(i + 1)
        for edge in topo.directionalized:
            # g.add_edge(edge.u + 1, edge.v + 1)
            g.add_edge(edge.u, edge.v)
        return g

    def _get_topo_dist(self, topo_graph: nx.DiGraph) -> np.ndarray:
        _inf = nx.number_of_nodes(topo_graph) + 5
        n = nx.number_of_nodes(topo_graph)
        dist = np.empty((n, n), dtype=np.int)
        dist[:, :] = _inf
        for u, v in topo_graph.edges:
            # if u == 0 or v == 0:
            #     continue
            # dist[u - 1][v - 1] = 1
            dist[u][v] = 1
        dist = _floyd(n, dist, _inf)
        return dist

    def _get_topo_mask(self, topo_graph: nx.DiGraph) -> torch.Tensor:
        n = nx.number_of_nodes(topo_graph)
        topo_mask = torch.zeros((n, n), dtype=torch.float)
        for u, v in topo_graph.edges:
            # if v == 0 or u == 0:
            #     continue
            # topo_mask[u - 1][v - 1] = 1.0
            topo_mask[u][v] = 1.0
        return topo_mask

    def _get_topo_edges(self, topo_graph: nx.DiGraph) -> np.ndarray:
        topo_edge = []
        for u, v in topo_graph.edges:
            # if v == 0 or u == 0:
            #     continue
            # topo_edge.append((u - 1, v - 1))
            topo_edge.append((u, v))
        return topo_edge

    def get_one(self) -> State:
        topo = self._cur_topo
        qubit_num = self.topo_qubit_num_map[topo.name]
        topo_graph = self.topo_graph_map[topo.name]
        topo_mask = self.topo_mask_map[topo.name]
        topo_dist = self._get_topo_dist(topo_graph=topo_graph)
        topo_edges = tuple(self.topo_edge_map[topo.name])

        min_gn = 2
        gate_num = randint(min_gn, max(self._max_gate_num, min_gn))
        success = False
        while not success:
            circ = Circuit(qubit_num)
            circ.random_append(
                gate_num,
                typelist=[
                    GateType.crz,
                ],
            )
            circ_state = CircuitState(circ=circ, max_gate_num=self._max_gate_num)
            logic2phy = [i for i in range(self.topo_qubit_num_map[topo.name])]
            circ_state.eager_exec(logic2phy=logic2phy, topo_graph=topo_graph)
            success = circ_state.count_gate() > 0

        circ_pyg_data = circ_state.to_pyg(logic2phy)

        state = State(
            circ_graph=circ_state,
            topo=topo,
            topo_mask=topo_mask,
            topo_graph=topo_graph,
            topo_dist=topo_dist,
            topo_edges=topo_edges,
            circ_pyg_data=circ_pyg_data,
            logic2phy=logic2phy,
        )
        return state
