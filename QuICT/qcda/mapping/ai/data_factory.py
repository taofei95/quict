import copy
import os
import os.path as osp
from random import choice, randint
from typing import Dict, List, Tuple, Union

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


# TODO: Add virtual node for topo graph


class CircuitState:
    """DAG Representation of a quantum circuit. A virtual node will be
    added with label 0.
    """

    def __init__(
        self, circ: Union[Circuit, CompositeGate, List[BasicGate]], max_gate_num: int
    ) -> None:
        self._max_gate_num = max_gate_num
        q = circ.width()
        if isinstance(circ, CircuitBased):
            self._gates: List[BasicGate] = circ.gates
        elif isinstance(circ, list):
            self._gates = circ
        else:
            raise TypeError(
                "circ argument only supports Circuit/CompositeGate/List[BasicGate]"
            )

        self._graph = nx.DiGraph()
        # self._graph.add_node(0)
        for gid in range(len(self._gates)):
            # self._graph.add_node(gid + 1)
            self._graph.add_node(gid)

        # v_node = 0
        occupied = [-1 for _ in range(q)]
        self._bit2gid: List[List[int]] = [[] for _ in range(q)]
        """Qubit to all gates on it.
        """
        for gid, gate in enumerate(self._gates):
            assert gate.controls + gate.targets == 2, "Only 2 bit gates are supported."
            a, b = gate.cargs + gate.targs
            # Position to Gate ID
            self._bit2gid[a].append(gid)
            self._bit2gid[b].append(gid)
            # DAG edges
            if occupied[a] != -1:
                self._graph.add_edge(occupied[a], gid)
            if occupied[b] != -1:
                self._graph.add_edge(occupied[b], gid)
            occupied[a] = gid
            occupied[b] = gid
            # Virtual node edges
            # self._graph.add_edge(v_node, gid + 1)
            # self._graph.add_edge(gid + 1, v_node)

    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result._max_gate_num = self._max_gate_num
        result._graph = copy.deepcopy(self._graph)
        result._gates = copy.deepcopy(self._gates)
        # result._gates = self._gates
        result._bit2gid = copy.deepcopy(self._bit2gid)
        return result

    def count_gate(self) -> int:
        return nx.number_of_nodes(self._graph)

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
        bit = 0
        while bit < len(self._bit2gid):
            if not self._bit2gid[bit]:
                bit += 1
                continue
            front = self._bit2gid[bit][0]
            gate = self._gates[front]
            a, b = gate.cargs + gate.targs
            if (
                (not self._bit2gid[a])
                or (not self._bit2gid[b])
                or self._bit2gid[a][0] != self._bit2gid[b][0]
            ):
                # Cannot remove anymore. Step forward.
                bit += 1
                continue
            _a, _b = logic2phy[a], logic2phy[b]
            if topo_graph.has_edge(_a, _b):
                # This gate can be removed
                self._bit2gid[a].pop(0)
                self._bit2gid[b].pop(0)
                gid = front
                self._graph.remove_node(gid)
                if physical_circ is not None:
                    with physical_circ:
                        gate & [_a, _b]
                remove_cnt += 1
                # There may be more gate can be removed after removal.
                # Do not add bit and keep exploring this bit.
            else:
                bit += 1

        return remove_cnt

    def sample_bias(
        self,
        topo_dist: np.ndarray,
        cur_logic2phy: List[int],
        next_logic2phy: List[int],
        qubit_number: int,
    ) -> float:
        """Summation of topological distances of all first gates on each qubits.

        Args:
            topo_dist (np.ndarray): Physical device topology distance
            cur_logic2phy (List[int]): Current logical to physical mapping
            next_logic2phy (List[int]): Next logical to physical mapping
            qubit_number (int): Number of qubit in physical layout.

        Returns:
            float: Bias based on distance summation
        """
        return 0.0
        # s = 0.0
        # for bit_stick in self._bit2gid:
        #     if not bit_stick:
        #         continue
        #     gate = self._gates[bit_stick[0]]
        #     a, b = gate.cargs + gate.targs
        #     _a, _b = cur_logic2phy[a], cur_logic2phy[b]
        #     prev_d = topo_dist[_a][_b]
        #     _a, _b = next_logic2phy[a], next_logic2phy[b]
        #     next_d = topo_dist[_a][_b]
        #     s += prev_d - next_d
        # if abs(s) < 1e-6:
        #     s += 0.1
        # # s = max(s, 0)
        # # if s < 0:
        # #     s = s * 2
        # s = s / (qubit_number**2)
        # return s

    def to_pyg(self, logic2phy: List[int]) -> Union[PygData, None]:
        """Convert current data into PyG Data according to current mapping.

        Arg:
            logic2phy (List[int]): Logical to physical qubit mapping.

        Returns:
            Union[PygData, None]: Return None if the circuit cannot be constructed to a graph. 
                Otherwise return PyG data. Each normal node will be assigned to 2 qubit ID (starting from 2).
                Virtual node will be labeled as (1, 1). Some nodes will be appended to graph to ensure alignment.
                Appended nodes will be labeled as (0, 0).
        """
        if len(self._graph.edges) == 0:
            return None
        x = torch.zeros(self._max_gate_num, 2, dtype=torch.long)
        for node in self._graph.nodes:
            # if node == 0:
            #     continue
            gid = node
            gate = self._gates[gid]
            a, b = gate.cargs + gate.targs
            x[gid][0] = logic2phy[a] + 1
            x[gid][1] = logic2phy[b] + 1
        edge_index = []
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
        topo_pyg_data: PygData,
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
        self.topo_pyg_data = topo_pyg_data
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
        max_qubit_num: int,
        max_gate_num: int,
        data_dir: str = None,
    ) -> None:
        if data_dir is None:
            data_dir = osp.dirname(osp.realpath(__file__))
            data_dir = osp.join(data_dir, "data")
        self._data_dir = data_dir
        self._topo_dir = osp.join(data_dir, "topo")

        self._max_qubit_num = max_qubit_num
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

    def get_topo_pyg(self, topo_graph: nx.DiGraph) -> PygData:
        topo_pyg_data = from_networkx(topo_graph)
        # x = 1 + torch.arange(self._max_qubit_num + 1, dtype=torch.long)
        x = 1 + torch.arange(self._max_qubit_num, dtype=torch.long)
        topo_pyg_data.x = x.unsqueeze(dim=-1)
        return topo_pyg_data

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
        for i in range(self._max_qubit_num):
            g.add_node(i)
            # g.add_node(i + 1)
        for edge in topo.directionalized:
            # g.add_edge(edge.u + 1, edge.v + 1)
            g.add_edge(edge.u, edge.v)
        return g

    def _get_topo_dist(self, topo_graph: nx.DiGraph) -> np.ndarray:
        _inf = nx.number_of_nodes(topo_graph) + 5
        n = self._max_qubit_num
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
        topo_mask = torch.zeros(
            (self._max_qubit_num, self._max_qubit_num), dtype=torch.float
        )
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

    def get_one(self, topo_name: str = None) -> State:
        if topo_name is None:
            topo_name: str = choice(self.topo_names)
        # TODO: Rotate topo
        topo_name: str = "ibmq_lima"
        topo = self.topo_map[topo_name]
        qubit_num = self.topo_qubit_num_map[topo_name]
        topo_graph = self.topo_graph_map[topo_name]
        topo_mask = self.topo_mask_map[topo_name]
        topo_dist = self._get_topo_dist(topo_graph=topo_graph)
        topo_edges = tuple(self.topo_edge_map[topo_name])
        circ = Circuit(qubit_num)

        min_gn = 10
        gate_num = randint(min_gn, max(self._max_gate_num, min_gn))
        circ.random_append(
            gate_num,
            typelist=[
                GateType.crz,
            ],
        )
        circ_graph = CircuitState(circ=circ, max_gate_num=self._max_gate_num)
        logic2phy = [i for i in range(self.topo_qubit_num_map[topo_name])]
        circ_pyg_data = circ_graph.to_pyg(logic2phy)

        topo_pyg_data = self.get_topo_pyg(topo_graph=topo_graph)

        state = State(
            circ_graph=circ_graph,
            topo=topo,
            topo_mask=topo_mask,
            topo_graph=topo_graph,
            topo_dist=topo_dist,
            topo_edges=topo_edges,
            circ_pyg_data=circ_pyg_data,
            topo_pyg_data=topo_pyg_data,
            logic2phy=logic2phy,
        )
        return state
