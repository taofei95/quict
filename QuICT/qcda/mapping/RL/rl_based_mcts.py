#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/13 20:20:17
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   rl_based_mcts.py

from networkx.algorithms.matching import matching_dict_to_set
from networkx.exception import NodeNotFound
from networkx.readwrite.gml import parse_gml_lines
import torch 
import torch.nn as nn

from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection

from QuICT.qcda.mapping.table_based_mcts import *
from .state_agent import StateAgent
from .nn_model import TransformerU2GNN
from .experience_pool_v4 import ExperiencePool


class RLBasedMCTS(TableBasedMCTS):
    def __init__(self, model: nn.Module = None, device: torch.device = torch.device('cpu') ,play_times: int = 1, gamma: float = 0.7, Gsim: int = 50, size_threshold: int = 150, 
                 Nsim: int = 500, selection_times: int = 40 , c: int = 5, mode: MCTSMode = MCTSMode.TRAIN, 
                experience_pool: ExperiencePool = None, coupling_graph: CouplingGraph = None,
                input: Queue = None, output: Connection = None, id: int = 0, **params):

        super().__init__(play_times = play_times, selection_times = selection_times, gamma = gamma, Gsim = Gsim, size_threshold = size_threshold, 
                        coupling_graph = coupling_graph, Nsim = Nsim, c = c, mode = mode, experience_pool = experience_pool)

        self._input = input
        self._output = output
        self._id = id
        self._device = device
        self._softmax = nn.Softmax(dim = 1)
        if self._mode == MCTSMode.EVALUATE:
            self._model = model

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
        
        
        
        self._add_initial_single_qubit_gate(node = self._root_node)
        self._add_executable_gates(node = self._root_node)
        
        self._expand(node = self._root_node)
        num_of_swap_gates = 0
        while self._root_node.is_terminal_node() is not True:
            
            self._search(root_node = self._root_node)
            node = self._root_node
            
            if self._mode == MCTSMode.TRAIN:
                if self._experience_pool is not None:
                    self._transform_to_training_data(node = node, experience_pool = self._experience_pool)
                else:
                    raise Exception("Experience pool is not defined.")
            
            self._root_node = self._decide(node = self._root_node)
            num_of_swap_gates += 1
            self._physical_circuit.append(self._root_node.swap_of_edge)
            self._add_executable_gates(node = self._root_node)
            #print(self._root_node.num_of_gates)
    
        return self._logical_circuit_dag.size, num_of_swap_gates
    

    def _search(self, root_node: MCTSNode):
        for _ in range(self._selection_times):
            cur_node = self._select(node = root_node)
            if self._mode == MCTSMode.TRAIN:
                value = self._evaluate_for_traning(node = cur_node)
            elif self._mode == MCTSMode.EVALUATE:
                value = self._evaluate(node = cur_node)
            self._expand(node = cur_node)
            self._backpropagate(node = cur_node, value = value)
   
    # def _decide(self, node: MCTSNode)->MCTSNode:
    #     pass

    def _transform_to_training_data(self, node: MCTSNode, experience_pool: ExperiencePool):
        if node.state is None:
            super()._transform_to_training_data(node, experience_pool)
        else:
            adj, feature = node.state
            experience_pool.push(adj = adj, feature = feature, action_probability = node.edge_prob, value = node.value)

    def _evaluate(self, node: MCTSNode):
        adj, feature = self._get_data(node)
        adj, graph_pool, feature = transform_batch(batch_data = (adj, feature), device = self._device)
        policy_scores, value_scores = self._model(adj, graph_pool, feature)
        node.value = value_scores[0]
        extended_prob = self._softmax(policy_scores).detach().to(torch.device('cpu')).numpy().squeeze()
            #print(p)
        prob = np.array([  extended_prob[self._coupling_graph.edge_label(swap_gate)] 
                      for swap_gate in node.candidate_swap_list ])
        prob = prob / np.sum(prob)
        node.edge_prob = prob
        return value_scores


    def _evaluate_for_traning(self, node: MCTSNode):
        adj, feature = self._get_data(node)
        self._input.put((self._id, adj, feature))

        value, probability = self._output.recv()
        node.value = value
        node.edge_prob = probability
        return value

    def _get_data(self, node: MCTSNode):
        num_of_gates = self._size_threshold
       
        num_of_gates, state = self._circuit_dag.get_subcircuit(front_layer = node.front_layer, num_of_gates = num_of_gates)
        qubit_mapping = np.zeros(len(node.cur_mapping) + 1, dtype = np.int32) -1  
        qubit_mapping[0:-1] = np.array(node.cur_mapping)
       
        adj = np.apply_along_axis(self._fill_adj, axis = 1, arr = np.concatenate([np.arange(0, num_of_gates )[:,np.newaxis], state[0:num_of_gates,0:5]],axis = 1))
        qubit_indices = qubit_mapping[state[0:num_of_gates,5:]] + 1
        feature = self._coupling_graph.node_feature[qubit_indices,:].reshape(num_of_gates, -1)
        
        return adj, feature

     
    def random_simulate(self, logical_circuit: Circuit, init_mapping: List[int]):    
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
      
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        
        
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 

        cur_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy(), parent =None)
        
        num_of_swap_gates = 0
        while not cur_node.is_terminal_node():
            adj, feature = self._get_data(cur_node)
            adj, graph_pool, feature = transform_batch(batch_data = (adj, feature), device = self._device)
            
            policy_scores, _ = self._model(adj, graph_pool, feature)
            
            extended_prob = self._softmax(policy_scores).detach().to(torch.device('cpu')).numpy().squeeze()
            #print(p)
            prob = np.array([  extended_prob[self._coupling_graph.edge_label(swap_gate)] 
                       for swap_gate in cur_node.candidate_swap_list ])
            
            prob = prob / np.sum(prob)
            # prob = np.ones(len(cur_node.candidate_swap_list) , dtype = np.float32)
            
            # prob = prob / np.sum(prob)
           # print(prob)
            swap_idx = np.random.choice(len(prob), p = prob)
            #swap_idx = np.argmax(prob)
            cur_node.update_by_swap_gate(swap = cur_node.candidate_swap_list[swap_idx])
            num_of_swap_gates += 1
            print(cur_node.num_of_gates)

        return num_of_swap_gates


    def extended_random_simulate(self, logical_circuit: Circuit, init_mapping: List[int]):    
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
      
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        
        
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 

        cur_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy(), parent =None)
        
        num_of_swap_gates = 0
        while not cur_node.is_terminal_node():
            adj, feature = self._get_data(cur_node)
            adj, graph_pool, feature = transform_batch(batch_data = (adj, feature), device = self._device)
            
            policy_scores, _ = self._model(adj, graph_pool, feature)
            
            p = self._softmax(policy_scores).detach().to(torch.device('cpu')).numpy().squeeze()
            #print(p)
            swap_idx = np.random.choice(p.shape[0], p = p)
            swap_gate = self._coupling_graph.get_swap_gate(swap_idx)
            cur_node.update_by_swap_gate(swap = swap_gate)
            num_of_swap_gates += 1
            #print(cur_node.num_of_gates)

        return num_of_swap_gates


        
        



