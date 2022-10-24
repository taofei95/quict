from random import randint
from typing import List, Optional, Tuple

import networkx as nx
import torch
from QuICT.core import *
from QuICT.core.gate import CompositeGate, GateType
from torch_geometric.data import Batch as PygBatch
from torch_geometric.data import Data as PygData

from ..common.circuit_info import CircuitInfo as CircuitInfoBase
from ..common.data_factory import DataFactory as DataFactoryBase
from ..common.layout_info import LayoutInfo as LayoutInfoBase


class DataFactory(DataFactoryBase):
    def get_one(self) -> "State":
        topo = self._cur_topo
        qubit_num = self.topo_qubit_num_map[topo.name]
        topo_graph = self.topo_graph_map[topo.name]

        min_gn = self._max_gate_num // 3
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
            circ_state = CircuitInfo(circ=circ, max_gate_num=self._max_gate_num)
            logic2phy = [i for i in range(self.topo_qubit_num_map[topo.name])]
            circ_state.eager_exec(logic2phy=logic2phy, topo_graph=topo_graph)
            success = circ_state.count_gate() > 0

        state = State(
            circ_info=circ_state,
            topo=topo,
            logic2phy=logic2phy,
        )
        return state


class CircuitInfo(CircuitInfoBase):
    """DAG Representation of a quantum circuit. A virtual node will be
    added with label 0.
    """

    def layered_gate_matrices(self, logic2phy: List[int]) -> torch.Tensor:
        layers: List[List[Tuple[int, int]]] = []
        occupied = [-1 for _ in range(self._qubit_num)]
        for gid in sorted(self._graph.nodes):
            gate = self._gates[gid]
            a, b = gate.cargs + gate.targs
            layer_id = 1 + max(occupied[a], occupied[b])
            while len(layers) <= layer_id:
                layers.append([])
            occupied[a] = layer_id
            occupied[b] = layer_id
            _a, _b = logic2phy[a], logic2phy[b]
            layers[layer_id].append((_a, _b))
        q = self._qubit_num
        ans = torch.zeros(len(layers), q, q, dtype=torch.float)
        for idx, layer in enumerate(layers):
            for u, v in layer:
                ans[idx][u][v] = 1
                ans[idx][v][u] = 1
        ans = ans.view(-1, q * q)
        return ans

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
        g = nx.convert_node_labels_to_integers(self._graph)
        # g = self._graph
        for node in g.nodes:
            gid = g.nodes[node]["gid"]
            gate = self._gates[gid]
            a, b = gate.cargs + gate.targs
            x[node][0] = logic2phy[a] + 1
            x[node][1] = logic2phy[b] + 1
            # Add self loops
            edge_index.append([gid, gid])
        for u, v in g.edges:
            edge_index.append([u, v])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
        data = PygData(x=x, edge_index=edge_index)
        return data


class TopoInfo(LayoutInfoBase):
    pass


class State:
    def __init__(
        self,
        circ_info: CircuitInfo,
        topo: Layout,
        logic2phy: List[int],
        phy2logic: Optional[List[int]] = None,
    ) -> None:
        self.circ_info = circ_info
        self.topo_info = TopoInfo(topo=topo)
        self.logic2phy = logic2phy
        """Logical to physical mapping
        """
        q = topo.qubit_number
        self.phy2logic = phy2logic
        if self.phy2logic is None:
            self.phy2logic = [0 for _ in range(q)]
        for i in range(q):
            self.phy2logic[self.logic2phy[i]] = i

        self._circ_pyg_data = None
        self._circ_layered_matrices = None

    @property
    def circ_pyg_data(self):
        if self._circ_pyg_data is None:
            self._circ_pyg_data = self.circ_info.to_pyg(self.logic2phy)
        return self._circ_pyg_data

    @property
    def circ_layered_matrices(self) -> torch.Tensor:
        if self._circ_layered_matrices is None:
            self._circ_layered_matrices = self.circ_info.layered_gate_matrices(
                self.logic2phy
            )
        return self._circ_layered_matrices

    def remained_circ(self) -> CompositeGate:
        return self.circ_info.remained_circ(self.logic2phy)

    def biased_random_swap(self) -> Tuple[int, int]:
        return self.circ_info.biased_random_swap(
            self.topo_info.topo_dist, self.logic2phy
        )

    def to_nn_data(self):
        return self.circ_pyg_data

    @staticmethod
    def batch_from_list(data_list: list, device: str):
        return PygBatch.from_data_list(data_list=data_list).to(device=device)


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
