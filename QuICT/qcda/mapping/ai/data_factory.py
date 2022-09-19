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


class CircuitGraph:
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
        for gid in range(len(self._gates)):
            self._graph.add_node(gid + 1)

        v_node = 0
        occupied = [-1 for _ in range(q)]
        self._bit2gid: List[List[int]] = []
        """Qubit to all gates on it.
        """
        for gid, gate in enumerate(self._gates):
            assert gate.controls + gate.targets == 2, "Only 2 bit gates are supported."
            args = tuple(gate.cargs + gate.targs)
            # Position to Gate ID
            self._bit2gid[args[0]].append(gid)
            self._bit2gid[args[1]].append(gid)
            # DAG edges
            if occupied[args[0]] != -1:
                self._graph.add_edge(occupied[args[0]], gid)
            if occupied[args[1]] != -1:
                self._graph.add_edge(occupied[args[1]], gid)
            occupied[args[0]] = gid
            occupied[args[1]] = gid
            # Virtual node edges
            self._graph.add_edge(v_node, gid)
            self._graph.add_edge(gid, v_node)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

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
        return nx.number_of_nodes(self._graph) - 1

    def eager_exec(self, logic2phy: List[int], topo_graph: nx.Graph) -> int:
        """Eagerly remove all executable gates for now.

        Args:
            logic2phy (List[int]): Current logical to physical qubit mapping.
            topo_graph (nx.Graph): Physical topology graph.

        Returns:
            int: Removed gate number.
        """
        bit = 0
        while bit <= len(self._bit2gid):
            if not self._bit2gid[bit]:
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
            if topo_graph.has_edge(_a, _b) or topo_graph.has_edge(_b, _a):
                # This gate can be removed
                self._bit2gid[a].pop(0)
                self._bit2gid[b].pop(0)
                gid = front
                self._graph.remove_node(gid + 1)
            # There may be more gate can be removed after removal.
            # Do not add bit and keep exploring this bit.

    def remove_gate(self, logical_pos: Tuple[int, int]) -> bool:
        """Remove 2-bit gate at give position.

        Args:
            logical_pos (Tuple[int, int]): Gate to be removed. Position is the qubit targets in logical view.
                The order does not matter.

        Returns:
            bool: Whether removal operation successes.
        """
        u, v = logical_pos
        if (
            (not self._bit2gid[u])  # No gate on qubit u
            or (not self._bit2gid[v])  # No gate on qubit v
            or (
                self._bit2gid[u][0] != self._bit2gid[v][0]
            )  # Gates on (u, v) are not the same
        ):
            return False
        gid = self._bit2gid[u][0]
        self._bit2gid[u].pop(0)
        self._bit2gid[v].pop(0)
        self._graph.remove_node(gid + 1)
        return True

    def to_pyg(self, logic2phy: List[int]) -> PygData:
        """Convert current data into PyG Data according to current mapping.

        Arg:
            logic2phy (List[int]): Logical to physical qubit mapping.

        Returns:
            PygData: PyG data. Each normal node will be assigned to 2 qubit ID (starting from 2).
                Virtual node will be labeled as (1, 1). Some nodes will be appended to graph to ensure alignment.
                Appended nodes will be labeled as (0, 0).
        """
        x = torch.zeros(self._max_gate_num + 1, 2, dtype=torch.long)
        x[0][0] = 1
        x[0][1] = 1
        for node in self._graph.nodes:
            gid = int(node["gid"])
            gate = self._gates[gid]
            args = gate.cargs + gate.targs
            x[gid + 1][0] = logic2phy[args[0]] + 2
            x[gid + 1][1] = logic2phy[args[1]] + 2
        edge_index = []
        for u, v in self._graph.edges:
            edge_index.append([u, v])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
        data = PygData(x=x, edge_index=edge_index)
        return data


class State:
    def __init__(
        self,
        circ_graph: CircuitGraph,
        topo: Layout,
        topo_mask: torch.Tensor,
        topo_graph: nx.Graph,
        topo_dist: np.ndarray,
        topo_edges: Tuple[Tuple[int, int]],
        pyg_data: PygData,
        logic2phy: List[int],
        phy2logic: List[int] = None,
    ) -> None:
        self.circ_graph = circ_graph
        self.topo = topo
        self.topo_mask = topo_mask
        self.topo_graph = topo_graph
        self.topo_dist = topo_dist
        self.topo_edges = topo_edges
        self.pyg_data = pyg_data
        self.logic2phy = logic2phy
        """Logical to physical mapping
        """
        self.phy2logic = phy2logic
        """Physical to logical mapping
        """
        if phy2logic is None:
            self.phy2logic = [-1 for _ in range(len(logic2phy))]
            for i in range(len(logic2phy)):
                self.phy2logic[logic2phy[i]] = i


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
            topo_graph = self._get_topo_graph(topo)
            self._topo_graph_map[topo_name] = topo_graph
            self._topo_qubit_num_map[topo_name] = topo.qubit_number
            self._topo_dist_map[topo_name] = self._get_topo_dist(topo_graph=topo_graph)

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

    def _get_topo_graph(self, topo: Layout) -> nx.Graph:
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

    def _get_topo_dist(self, topo_graph: nx.Graph) -> np.ndarray:
        _inf = nx.number_of_nodes(topo_graph) + 5
        n = self._max_qubit_num
        dist = np.empty((n, n), dtype=np.int)
        dist[:, :] = _inf
        for u, v in topo_graph.edges:
            dist[u][v] = 1
            dist[v][u] = 1
        dist = _floyd(n, dist, _inf)
        return dist

    def _get_topo_mask(self, topo_graph: nx.Graph) -> torch.Tensor:
        topo_mask = torch.zeros(
            (self._max_qubit_num, self._max_qubit_num), dtype=torch.float
        )
        for u, v in topo_graph.edges:
            topo_mask[u][v] = 1.0
            topo_mask[v][u] = 1.0
        return topo_mask

    def _get_topo_edges(self, topo_graph: nx.Graph) -> np.ndarray:
        topo_edge = []
        for u, v in topo_graph.edges:
            topo_edge.append((u, v))
            topo_edge.append((v, u))
        return topo_edge

    def get_one(self, topo_name: str = None) -> State:
        if topo_name is None:
            topo_name: str = choice(self.topo_names)
        # topo_name: str = "ibmq_peekskill"
        topo = self.topo_map[topo_name]
        qubit_num = self.topo_qubit_num_map[topo_name]
        topo_graph = self.topo_graph_map[topo_name]
        topo_mask = self.topo_mask_map[topo_name]
        topo_dist = self._get_topo_dist(topo_graph=topo_graph)
        topo_edges = tuple(self.topo_edge_map[topo_name])
        circ = Circuit(qubit_num)

        min_gn = 80
        gate_num = randint(min_gn, max(self._max_gate_num, min_gn))
        circ.random_append(
            gate_num,
            typelist=[
                GateType.crz,
            ],
        )
        circ_graph = CircuitGraph(circ=circ, max_gate_num=self._max_gate_num)
        logic2phy = [i for i in range(self.topo_qubit_num_map[topo_name])]
        pyg_data = circ_graph.to_pyg(logic2phy)

        state = State(
            circ_graph=circ_graph,
            topo=topo,
            topo_mask=topo_mask,
            topo_graph=topo_graph,
            topo_dist=topo_dist,
            topo_edges=topo_edges,
            pyg_data=pyg_data,
            logic2phy=logic2phy,
        )
        return state
