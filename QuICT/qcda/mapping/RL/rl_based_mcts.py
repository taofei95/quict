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

    def _select_best_child(self, cur_node: MCTSNode):
        """
        Select the best child for expansion
        """
        pass

    def _get_best_child(self, node: MCTSNode):
        """
        Used in decide function
        """
        return super()._get_best_child(node)

    def _rollout(self, node: MCTSNode, method: str):
        # TODO: add experience for the roll out trajectory
        return super()._rollout(node, method)

    def _backpropagate(self, cur_node: MCTSNode):
        self._state_agent.backward(cur_node)

    def _process_state_from_node(self, cur_node: MCTSNode):
        """
        Transform the info in `cur_nod` into structured state info for state agent
        The state contains two numpy arrays: 
          - The edges index array with size (num_edges, 2)
          - The adjacent matrix where the i-th row contains the edge indices of the i-th node
        This is designed for GPU acceleration.
        """
        front_layers = cur_node.front_layer()
        DAG = cur_node.circuit
        cur_mapping = cur_node.layout
        front_layer = cur_node.front_layer
        adj_matrix = []
        edge_list = []
        for dag_node in front_layer:
            pass

        