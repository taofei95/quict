from math import log, sqrt
import random
from tkinter.messagebox import NO
from typing import Dict, Optional, Tuple, Union, List
from QuICT.core import *
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.qcda.mapping.mcts_refactor.common.layout_info import LayoutInfo
from .common import CircuitInfo
from __future__ import annotations
import copy


class MCTSTreeNode:
    def __init__(
        self,
        circuit_info: CircuitInfo,
        layout_info: LayoutInfo,
        parent: Optional[MCTSTreeNode] = None,
        logic2phy: Optional[List[int]] = None,
        phy2logic: Optional[List[int]] = None,
    ) -> None:
        self.circ_info = circuit_info
        self.layout_info = layout_info
        self.executed_circ = CompositeGate()
        self.children: Dict[Tuple[int, int], MCTSTreeNode] = []
        self.q_value: float = 0.0
        self.parent = parent
        self.transition_reward = 0.0
        self.visit_cnt: int = 1

        if logic2phy is None:
            logic2phy = [i for i in range(self.circ_info._qubit_num)]
        self.logic2phy = logic2phy

        if phy2logic is None:
            phy2logic = [-1 for _ in range(self.circ_info._qubit_num)]
            for idx, val in enumerate(self.logic2phy):
                phy2logic[val] = idx
        self.phy2logic = phy2logic

    def _update_mapping(self, action: Tuple[int, int]):
        u, v = action
        self.phy2logic[u], self.phy2logic[v] = self.phy2logic[v], self.phy2logic[u]
        lu, lv = self.phy2logic[u], self.phy2logic[v]
        self.logic2phy[lu], self.phy2logic[lv] = self.logic2phy[lv], self.logic2phy[lu]

    def _recommended_action(self) -> List[Tuple[int, int]]:
        ans = []
        relative_qubit = set()
        for gate in self.circ_info.first_layer_gates.values():
            if gate.controls + gate.targets == 1:
                continue
            a, b = gate.cargs + gate.targs
            relative_qubit.add(a)
            relative_qubit.add(b)
        for a, b in self.layout_info.topo_edges:
            if a in relative_qubit or b in relative_qubit:
                ans.append((a, b))
        return ans

    def _expand_one(self, action: Tuple[int, int]) -> MCTSTreeNode:
        successor = MCTSTreeNode(
            circuit_info=copy.deepcopy(self.circ_info),
            layout_info=self.layout_info,
            parent=self,
            logic2phy=copy.copy(self.logic2phy),
            phy2logic=copy.copy(self.phy2logic),
        )
        successor._update_mapping(action=action)
        reward = successor.circ_info.eager_exec(
            logic2phy=successor.logic2phy,
            topo_graph=successor.layout_info.topo_graph,
            physical_circ=self.executed_circ,
        )
        successor.transition_reward = reward
        return successor

    def select_child(self, c: float) -> MCTSTreeNode:
        self.visit_cnt += 1
        candidates = [child for child in self.children.values()]
        candidates.sort(
            key=lambda x: x.q_value
            + x.transition_reward
            + c * sqrt(log(self.visit_cnt) / x.visit_cnt),
            reverse=True,
        )
        return candidates[0]

    def expand(self):
        assert len(self.children) == 0, "Only leaf node can be expanded!"
        for action in self._recommended_action():
            child = self._expand_one(action=action)
            self.children[action] = child
        assert len(self.children) > 0, "After expansion, node must have successors!"

    def _biased_random_simulate(self, sim_gate_num: int, epsilon: float) -> float:
        remove_cnt = 0
        cur = self
        step = 0
        while remove_cnt < sim_gate_num:
            candidates = cur._recommended_action()
            impact_factors = []
            for u, v in candidates:
                cur_d, nxt_d = 0, 0
                for gate in self.circ_info.first_layer_gates.values():
                    a, b = gate.cargs + gate.targs
                    _a, _b = self.logic2phy[a], self.logic2phy[b]
                    cur_d += self.layout_info.topo_dist[_a][_b]
                    l2p = copy.copy(self.logic2phy)
                    lu, lv = self.phy2logic[u], self.phy2logic[v]
                    l2p[lu], l2p[lv] = l2p[lv], l2p[lu]
                    _a, _b = l2p[a], l2p[b]
                    nxt_d += self.layout_info.topo_dist[_a][_b]
                factor = cur_d - nxt_d
                if factor < 1e-6 and factor > -1e-6:
                    factor += epsilon
                elif factor < 0:
                    factor = 0
                impact_factors.append(factor)
            action = random.choices(population=candidates, weights=impact_factors, k=1)[
                0
            ]
            suc = cur._expand_one(action)
            remove_cnt += suc.transition_reward
            cur = suc
            step += 1
        return float(step)

    def simulate(self, sim_cnt: int, sim_gate_num: int, gamma: float, epsilon: float):
        assert len(self.children) == 0, "Only leaf node can be simulated!"
        # Set a large initial value
        best = float(
            10 * self.circ_info.count_gate() * self.layout_info.layout.qubit_number
        )
        for _ in range(sim_cnt):
            cur = self._biased_random_simulate(sim_gate_num, epsilon)
            best = min(best, cur)
        sim_res = gamma ** (best / 2) * float(sim_gate_num)
        self.q_value = sim_res

    def back_propagate(self, gamma: float):
        parent = self.parent
        cur = self
        while parent is not None:
            parent.q_value = max(
                parent.q_value, gamma * (self.transition_reward + cur.q_value)
            )
            cur = parent
            parent = parent.parent

    def best_child(self) -> Tuple[Tuple[int, int], MCTSTreeNode, bool]:
        """Best child's action, state, and terminate info.

        Returns:
            Tuple[Tuple[int, int], MCTSTreeNode, bool]: Action to child, child node, whether mapping ends.
        """
        candidates = [pair for pair in self.children.items()]
        candidates.sort(
            key=lambda x: x[1].q_value + x[1].transition_reward, reverse=True
        )
        action, child = candidates[0]
        terminated = child.is_terminated_node()
        return action, child, terminated

    def is_terminated_node(self) -> bool:
        return self.circ_info.count_gate() == 0
