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
        Transform the info in `cur_node` into structured state info for state agent
        The state contains two numpy arrays: 
          - The edges index array with size (num_edges, 2, 2)
          - The adjacent matrix where the i-th row contains the edge indices of the i-th node
        This is designed for GPU acceleration.
        """
        DAG = cur_node.circuit_in_dag
        cur_mapping = cur_node.cur_mapping

        adj_matrix = []
        edge_list = []

        cur_layer_nodes = set(cur_node.front_layer)
        next_layer_nodes = set()

        edge_cnt = 0
        node_cnt = 0
        temp_node_mapping = {}

        while cur_layer_nodes:
            for node in cur_layer_nodes:
                adj_matrix.append([])
                temp_node_mapping[node] = node_cnt
                for suc_node in DAG.get_successor_nodes(node):

                    def get_physical_bits_of_gate(gate):
                        l_qubits = gate.cargs+gate.cargs
                        return [cur_mapping[qbit] for qbit in l_qubits]

                    edge_list.append([get_physical_bits_of_gate(node["gate"]),\
                                    get_physical_bits_of_gate(suc_node["gate"])])
                    adj_matrix[node_cnt].append(edge_cnt)
                    edge_cnt += 1

                next_layer_nodes = next_layer_nodes.union(DAG.get_successor_nodes(node))
            cur_layer_nodes = next_layer_nodes

            node_cnt += 1
            

        