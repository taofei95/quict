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
        TODO: remove 
        """
        pass

    def _get_best_child(self, node: MCTSNode):
        """
        Used in decide function
        TODO: remove 
        """
        return super()._get_best_child(node)

    def _expand(self, node: MCTSNode):
        """
        TODO: add NN evaluation to the function
        """
        pass

    def _rollout(self, node: MCTSNode, method: str):
        """
        TODO: add experience for the roll out trajectory
        """
        return super()._rollout(node, method)

    def _backpropagate(self, cur_node: MCTSNode):
        self._state_agent.backward(cur_node)
   
    def _process_state_from_node(self, cur_node: MCTSNode):
        """
        Transform the info in `cur_node` into structured state info for state agent.

        The state contains two numpy arrays: 
          - The edges index array with size (num_edges, 4)
          - The adjacent matrix where the i-th row contains the edge indices of the i-th node
        
        This is designed for GPU acceleration.
        """
        DAG = cur_node.circuit_in_dag
        cur_mapping = cur_node.cur_mapping

        adj_matrix = []
        edge_list = []
                
        cur_layer_nodes = set(cur_node.front_layer)
        visited_node = set(cur_layer_nodes)
        next_layer_nodes = set()

        edge_cnt = 0
        node_cnt = 0
        while cur_layer_nodes:
            for node in cur_layer_nodes:

                # Filter out single-qubit gate in front layers
                if node['gate'].is_single():
                    next_layer_nodes = next_layer_nodes.union(DAG.get_successor_nodes(node))
                    continue

                # Filter out single-qubit gates in the successor nodes
                suc_node_list = []
                for suc_node in DAG.get_successor_nodes(node):
                    while suc_node['gate'].is_single():
                        if not DAG.get_successor_nodes(suc_node):
                            break
                        suc_node = DAG.get_successor_nodes(suc_node)[0]

                    if not suc_node['gate'].is_single():
                        suc_node_list.append(suc_node)

                adj_matrix.append([])
                for suc_node in suc_node_list:
                    if suc_node in visited_node:
                        continue

                    def get_physical_bits_of_gate(gate):
                        l_qubits = gate.cargs+gate.cargs
                        return [cur_mapping[qbit] for qbit in l_qubits]

                    edge_list.append([get_physical_bits_of_gate(node["gate"]),\
                                    get_physical_bits_of_gate(suc_node["gate"])])
                    adj_matrix[node_cnt].append(edge_cnt)
                    edge_cnt += 1
                    next_layer_nodes.add(suc_node)
                    visited_node.add(suc_node)

            cur_layer_nodes = next_layer_nodes

            node_cnt += 1
            

        