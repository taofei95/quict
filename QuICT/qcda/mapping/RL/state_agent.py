#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:52:53
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   state_agent.py

from .._mcts_base import MCTSNode
class StateAgent(object):
    def __init__(self) -> None:
        super().__init__()

    def backpropagate(self, node: MCTSNode):
        pass


    def get_state(self, node: MCTSNode):
        pass