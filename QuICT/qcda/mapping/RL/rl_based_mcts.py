#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/13 20:20:17
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   rl_based_mcts.py

from ..table_based_mcts import *
from .state_agent import StateAgent

class RLBasedMCTS(TableBasedMCTS):
    def __init__(self):
        self._state_agent = StateAgent()

    def _get_best_child(self, cur_node: MCTSNode):
        pass

    def _backpropagate(self, cur_node: MCTSNode):
        self._state_agent.backward(cur_node)