#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/13 20:20:17
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   rl_based_mcts.py

import torch 
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet

from torch.multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection

from  QuICT.qcda.mapping.table_based_mcts import *
from RL.experience_pool_v4 import ExperiencePool



class OracleBasedMCTS(TableBasedMCTS):
    def _upper_confidence_bound(self, node: MCTSNode)->float:
        """
        The upper confidence bound of the node
        """
        # if node.visit_count == 0:
        #     return node.reward + node.value + self._c*100
        # else:
        #     return node.reward + node.value + self._c* np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
        #return  1- (node.value)/(node.sim_gates+0.001)  + self._c* np.sqrt(np.log(node.parent.visit_count) / (node.visit_count+0.001))
        return float(node.sim_gates + node.reward)/float(node.value + 1)  + self._c* np.sqrt(np.log(node.parent.visit_count) / (node.visit_count+0.001))
    def _upper_confidence_bound_with_predictor(self, node: MCTSNode)->float:
        """
        The upper confidence bound with predictor of the node
        """
        return   float(node.sim_gates + node.reward)/float(node.value + 1)  + self._c* node.prob_of_edge *np.sqrt(node.parent.visit_count) / (node.visit_count+1) 


    def __init__(self, play_times: int = 1, gamma: float = 0.7, Gsim: int = 150, size_threshold: int = 150, 
                 Nsim: int = 500, selection_times: int = 40 , c: int = 10, mode: MCTSMode = MCTSMode.SEARCH, rl: RLMode = RLMode.WARMUP,
                experience_pool: ExperiencePool = None, coupling_graph: CouplingGraph = None, log_path: str = None, extended: bool = False, **params):

        super().__init__(play_times = play_times, selection_times = selection_times, gamma = gamma, Gsim = Gsim, size_threshold = size_threshold, extended = extended,
                        coupling_graph = coupling_graph, Nsim = Nsim, c = c, mode = mode, experience_pool = experience_pool, log_path = log_path, rl = rl)

        self._sim_method = 1

   
    def search(self, logical_circuit: Circuit = None, init_mapping: List[int] = None):
        self._num_of_executable_gate = 0
        self._logical_circuit_dag = DAG(circuit = logical_circuit, mode = Mode.WHOLE_CIRCUIT) 
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
      
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        
        
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 

        self._root_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy(), parent =None)
        
        self._root_node.sim_gates = min(self._Gsim, self._root_node.num_of_gates)
        self._create_random_simulator()
        self._add_initial_single_qubit_gate(node = self._root_node)
        self._add_executable_gates(node = self._root_node)
        
        #self._expand(node = self._root_node)
        self._num_of_swap_gates = 0
        while self._root_node.is_terminal_node() is not True:
            
            self._search(root_node = self._root_node)
            node = self._root_node
            print([self._num_of_executable_gate, self._num_of_swap_gates ])
            print(self._root_node.value)
            print(self._root_node.visit_count_prob)
            self._logger.info(self._root_node.value)
            self._logger.info([self._root_node.num_of_gates, self._num_of_swap_gates ])
            
            self._root_node = self._decide(node = self._root_node)
            if self._root_node.reward == 0:
                self._fallback_count +=1
            else:
                self._fallback_count = 0

            if self._mode == MCTSMode.TRAIN:
                if self._experience_pool is not None:
                    #print("exp")
                    self._transform_to_training_data(node = node)
                else:
                    raise Exception("Experience pool is not defined.")

            self._num_of_swap_gates += 1
            self._physical_circuit.append(self._coupling_graph.get_swap_gate(self._root_node.swap_of_edge))
            self._add_executable_gates(node = self._root_node)
            #print(self._root_node.num_of_gates)
            if self._fallback_count > self._fallback_threshold:
                self._root_node = self._fall_back(node = self._root_node)

        if self._mode == MCTSMode.TRAIN:     
            if self._experience_pool is not None:
                circuit_size = np.array(self._num_list)
                label = np.array(self._label_list)
                adj = np.array(self._adj_list)
                qubits = np.array(self._qubits_list)
                action_probability = np.array(self._action_probability_list)
                value = np.array(self._value_list)
                #print(value)
                self._experience_pool.extend(adj = adj, qubits = qubits, action_probability = action_probability, value = value, circuit_size = circuit_size, swap_label = label, num = len(self._adj_list))
            else:
                raise Exception("Experience pool is not defined.")
        
        return self._logical_circuit_dag.size, self._num_of_swap_gates
    

    def _search(self, root_node: MCTSNode):
        for _ in range(self._selection_times):
            cur_node = self._select(node = root_node)
            value = self._rollout(node = cur_node) 
            self._expand(node = cur_node)
            self._backpropagate(node = cur_node, value = value)
   
    # def _decide(self, node: MCTSNode)->MCTSNode:
    #     pass



    def _decide(self,node : MCTSNode)-> MCTSNode:
        """
        Decide which child node to move into 
        """
        node = self._get_best_child(node)
        node.parent = None
        #node.clear()
        node.sim_gates = min(self._Gsim, node.num_of_gates)
        if node.sim_gates <=0:
            node.sim_gates = min(self._Gsim, node.num_of_gates)
            node.clear()
        return node

    def _rollout(self, node: MCTSNode )-> float:
        """
        Do a heuristic search for the sub circuit with Gsim gates from the current node by the specified method
        """
        res = 0
        front_layer = [ self._circuit_dag.index[i]  for i in node.front_layer ]
        qubit_mask =  [ self._circuit_dag.index[i] if i != -1 else -1 for i in node.qubit_mask ]
            # for i in range(self._Nsim):
            #     N = min(N, self._random_simulation(node))
            #     print("%d and %d"%(N,node.num_of_gates))

        res = self._rs.simulate(front_layer = front_layer, qubit_mapping = node.cur_mapping, qubit_mask = qubit_mask, num_of_subcircuit_gates = node.sim_gates, num_of_iterations = self._Nsim ,simulation_mode = self._sim_method)
        #print(N)
            #res = np.float_power(self._gamma, N/2) * float(self._Gsim)
            #print(list(front_layer))
            #print(list(qubit_mask))
            #print("%d and %d"%(N,node.num_of_gates))

        node.sim_value = res
        node.value  =  res
        node.w += res

        return res

    
    def _backpropagate(self, node : MCTSNode, value: int):
        """
        Use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        
        cur_node = node
        bp_value = value + 1
        
        while cur_node.parent is not None:
            if bp_value < cur_node.parent.value:
                cur_node.parent.value = bp_value
            bp_value = bp_value + 1
            cur_node = cur_node.parent
    

    def _expand(self, node: MCTSNode):
        """
        Open all child nodes of the current node by applying the swap gates in candidate swap list
        """
        if node is None:
            raise Exception("Node can't be None")
        if node.sim_gates>0:
            self._get_candidate_swap_list(node = node)
            for swap, prob in zip(node.candidate_swap_list, node.edge_prob):
                child_node = node.add_child_node_by_swap_gate(swap, prob)
                sim_gates = node.sim_gates - child_node.reward
                child_node.sim_gates = sim_gates if sim_gates > 0 else 0 
                child_node.value = child_node.num_of_gates
    

    def _get_best_child(self, node: MCTSNode)-> MCTSNode:
        """
        Get the child with highest score of the current node as the next root node. 
        """
        res_node = None
        score = -1
        idx = -1
        prob = np.ndarray(len(node.candidate_swap_list), dtype = np.float)
        value = np.ndarray(len(node.candidate_swap_list), dtype = np.int32)
        for i, child in enumerate(node.children):
            prob[i] = child.visit_count
            value[i] = child.value 
        print(prob)
        prob = prob / np.sum(prob)
    
        node.visit_count_prob = prob
    
        #idx = np.argmax(prob)
       
        idx = np.argmin(value)
        # for i, child in enumerate(node.children):
        #     # reward_list.append(child.reward)
        #     # value_list.append(child.value)
        #     if child.value  > score:
        #         res_node = child
        #         score = res_node.value  
        #         idx = i   
        # # print(reward_list)
        # # print(value_list) 
        res_node = node.children[idx]
        node.best_swap_gate = node.candidate_swap_list[idx]
        return res_node
    
    def _select(self, node: MCTSNode)-> MCTSNode:
        """
        Select the child node with highest score to expand
        """
        cur_node = node 
        cur_node.visit_count = cur_node.visit_count + 1
        while cur_node.is_leaf_node() is not True:
            cur_node = self._select_next_child(cur_node)
            cur_node.visit_count = cur_node.visit_count + 1
        return cur_node

    def _select_next_child(self, node: MCTSNode)-> MCTSNode:
        """
        Select the next child to be expanded of the current node 
        """
        res_node = None
        score = float('-inf')
        for child in node.children:
            UCB = self._upper_confidence_bound(node = child)
            if UCB >score:
                res_node = child
                score = UCB
        return res_node

    
    def _get_data(self, node: MCTSNode):
       
        qubit_mask =  [  -1  for _ in range(len(node.cur_mapping))]
        for i, q in  enumerate(node.qubit_mask):
            qubit_mask[node.inverse_mapping[i]] = q
        #print(qubit_mask)
        num_of_gates, state = self._circuit_dag.get_subcircuit(front_layer = node.front_layer, qubit_mask = qubit_mask, num_of_gates = self._size_threshold)
    
        qubit_mapping = np.zeros(len(node.cur_mapping) + 1, dtype = np.int32) -1  
        qubit_mapping[0:-1] = np.array(node.cur_mapping)

        padding_mask = np.zeros(self._size_threshold, dtype = np.uint8)
        padding_mask[num_of_gates:] = 1
        adj = np.apply_along_axis(self._fill_adj, axis = 1, arr = np.concatenate([np.arange(0, self._size_threshold)[:,np.newaxis], state[:,0:5]],axis = 1))
        qubit_indices = qubit_mapping[state[:,5:]] 
        
        node.state = (adj, qubit_indices, num_of_gates)

        return qubit_indices, padding_mask, adj

    


       


        
        



