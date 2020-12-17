#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/13 20:20:17
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   rl_based_mcts.py

from .._mcts_base import *
from .state_agent import StateAgent

class RLBasedMCTS(MCTSBase):
    def __init__(self):
        self._state_agent = StateAgent()

    def _select(self, cur_node: MCTSNode):
        return super()._select(cur_node)

    def _expand(self, cur_node: MCTSNode):
        return super()._expand(cur_node)

    def _backpropagate(self, cur_node: MCTSNode):
        return super()._backpropagate(cur_node)