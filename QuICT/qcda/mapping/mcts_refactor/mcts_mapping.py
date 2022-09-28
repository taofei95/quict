from typing import Union
from QuICT.core import *
from QuICT.core.gate import CompositeGate
from QuICT.core.gate.gate import *
from .common.layout_info import LayoutInfo
from .mcts_tree import MCTSTree


class MCTSMapping:
    def __init__(self, layout: Layout) -> None:
        self._layout_info = LayoutInfo(layout=layout)

    def execute(self, circuit_like: Union[Circuit, CompositeGate]) -> CompositeGate:
        mcts_tree = MCTSTree(circuit_like=circuit_like, layout_info=self._layout_info)
        cg = CompositeGate()
        terminated = False
        while not terminated:
            action, part, terminated = mcts_tree.step()
            with cg:
                Swap & action
                part & action
        mcts_tree = None
        return cg
