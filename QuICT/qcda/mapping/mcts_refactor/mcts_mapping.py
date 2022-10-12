from typing import List, Tuple, Union
from QuICT.core import *
from QuICT.core.gate import CompositeGate
from QuICT.core.gate.gate import *
from .common import CircuitInfo, LayoutInfo

from .mcts_tree import MCTSTree


class MCTSMapping:
    def __init__(
        self,
        layout: Layout,
        bp_num: int = 10,
        sim_cnt: int = 10,
        sim_gate_num: int = 30,
        gamma: float = 0.9,
        epsilon: float = 0.001,
        c: float = 0.01,
    ) -> None:
        self._layout_info = LayoutInfo(layout=layout)
        self._bp_num = bp_num
        self._sim_cnt = sim_cnt
        self._sim_gate_num = sim_gate_num
        self._c = c
        self._gamma = gamma
        self._epsilon = epsilon

    def execute(
        self,
        circuit_like: Union[Circuit, CompositeGate],
        with_final_mapping: bool = False,
    ) -> Union[
        Union[Circuit, CompositeGate],
        Tuple[Union[Circuit, CompositeGate], List[int], List[int]],
    ]:
        output_circ = type(circuit_like) is Circuit
        cg = CompositeGate()
        q = circuit_like.width()
        circuit_info = CircuitInfo(
            circ=circuit_like, max_gate_num=len(circuit_like.gates)
        )
        # Bypass first part
        circuit_info.eager_exec(
            logic2phy=[i for i in range(q)],
            physical_circ=cg,
            topo_graph=self._layout_info.topo_graph,
        )
        # Execute search on remain
        mcts_tree = MCTSTree(
            circuit_info=circuit_info,
            layout_info=self._layout_info,
            bp_num=self._bp_num,
            sim_cnt=self._sim_cnt,
            sim_gate_num=self._sim_gate_num,
            gamma=self._gamma,
            epsilon=self._epsilon,
            c=self._c,
        )

        terminated = mcts_tree._root.is_terminated_node()
        while not terminated:
            action, part, terminated = mcts_tree.step()
            with cg:
                Swap & list(action)
            cg.extend(part)
        final_logic2phy, final_phy2logic = (
            mcts_tree._root.logic2phy,
            mcts_tree._root.phy2logic,
        )
        del mcts_tree

        if output_circ:
            result = Circuit(cg.width())
            result.extend(cg.gates)
        else:
            result = cg

        if not with_final_mapping:
            return result
        else:
            return result, final_logic2phy, final_phy2logic
