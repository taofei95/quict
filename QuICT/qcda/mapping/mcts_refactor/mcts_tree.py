from typing import Tuple, Union
from .mcts_tree_node import MCTSTreeNode
from QuICT.core import *
from QuICT.core.gate import CompositeGate
from .common import LayoutInfo


class MCTSTree:
    def __init__(
        self,
        circuit_like: Union[Circuit, CompositeGate],
        layout_info: LayoutInfo,
        bp_num: int,
        sim_cnt: int,
        sim_gate_num: int,
        gamma: float,
        epsilon: float = 0.001,
        c: float = 0.1,
    ) -> None:
        self._root = MCTSTreeNode(circuit_like=circuit_like, parent=None)
        self._layout_info = layout_info

        self._bp_num = bp_num
        self._sim_cnt = sim_cnt
        self._sim_gate_num = sim_gate_num
        self._c = c
        self._gamma = gamma
        self._epsilon = epsilon

    def step(self) -> Tuple[Tuple[int, int], CompositeGate, bool]:
        """Execute multiple search rounds. Return a swap selection and update current search tree root.

        Returns:
            Tuple[Tuple[int, int], bool]: Selected swap action, whether mapping ends.
        """
        for _ in range(self._bp_num):
            cur = self._root
            while len(cur.children) > 0:
                cur = self._root.select_child(self._c)
            cur.expand()
            for leaf in cur.children.values():
                leaf.simulate(
                    sim_cnt=self._sim_cnt,
                    sim_gate_num=self._sim_gate_num,
                    gamma=self._gamma,
                    epsilon=self._epsilon,
                )
                leaf.back_propagate(self._gamma)
        action, self._root, terminated = self._root.best_child()
        return action, self._root.executed_circ, terminated
