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
from .experience_pool_v4 import ExperiencePool



class RLBasedMCTS(TableBasedMCTS):
    def __init__(self, model: nn.Module = None, device: torch.device = torch.device('cpu') ,play_times: int = 1, gamma: float = 0.7, Gsim: int = 50, size_threshold: int = 150, 
                 Nsim: int = 500, selection_times: int = 40 , c: int = 10, mode: MCTSMode = MCTSMode.SEARCH, rl: RLMode = RLMode.WARMUP,
                experience_pool: ExperiencePool = None, coupling_graph: CouplingGraph = None, log_path: str = None,
                input: Queue = None, output: Connection = None, id: int = 0, extended: bool = False, **params):

        super().__init__(play_times = play_times, selection_times = selection_times, gamma = gamma, Gsim = Gsim, size_threshold = size_threshold, extended = extended,
                        coupling_graph = coupling_graph, Nsim = Nsim, c = c, mode = mode, experience_pool = experience_pool, log_path = log_path, rl = rl)

        self._tau = 1
        self._input = input
        self._output = output
        self._id = id
        self._device = device
        self._softmax = nn.Softmax(dim = 0)
        # print(self._mode)
        # print(self._mode.value)
        # print(MCTSMode.SEARCH.value)
        # print(self._mode.name)
        # print(MCTSMode.SEARCH.name)
        
        # print(repr(self._mode))
        # print(repr(MCTSMode.SEARCH))
        # print(self._mode == MCTSMode.SEARCH)
        # print(MCTSMode.SEARCH)
        if self._mode == MCTSMode.SEARCH or self._mode == MCTSMode.EVALUATE:
            print("model")
            if model is not None:
                self._model = model
                self._model.eval()
                print("model")
            else:
                raise Exception("No nn model")

   
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
        
        #self._expand(node = self._root_node)
        self._num_of_swap_gates = 0
        while self._root_node.is_terminal_node() is not True:
            
            self._search(root_node = self._root_node)
            node = self._root_node
            print([self._root_node.num_of_gates, self._num_of_swap_gates ])
            print(self._root_node.value)
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
            if self._mode == MCTSMode.TRAIN:
                value = self._evaluate_for_training(node = cur_node)
            else:
                value = self._evaluate(node = cur_node)
            self._expand(node = cur_node)
            self._backpropagate(node = cur_node, value = value)
   
    # def _decide(self, node: MCTSNode)->MCTSNode:
    #     pass

    def _transform_to_training_data(self, node: MCTSNode):
        if node.state is None:
            super()._transform_to_training_data(node, node.best_swap_gate)
        else:
            adj, qubits, num_of_gates = node.state
            self._label_list.append(node.best_swap_gate)
            self._num_list.append(num_of_gates)
            self._adj_list.append(adj)
            self._qubits_list.append(qubits)
            self._value_list.append(node.value)
            if self._extended:
                self._action_probability_list.append(node.visit_count_prob)
            else:
                self._action_probability_list.append(self._extend_action_probability(node.visit_count_prob, node.candidate_swap_list))


    def _evaluate(self, node: MCTSNode):
        qubits, padding_mask, adj = self._get_data(node)

        qubits, padding_mask, adj = transform_batch(batch_data = (qubits, padding_mask, adj), device = self._device)
        qubits, padding_mask, adj =  qubits[None, :, :], padding_mask[None, :], adj[None,:,:]
        policy_scores, value_scores = self._model(qubits, padding_mask, adj)
        value = value_scores.detach().to(torch.device('cpu')).numpy().squeeze()
        node.sim_value = value
        node.value =  value
        node.w = value
        prob = self._softmax(policy_scores.squeeze()).detach().to(torch.device('cpu')).numpy()
        
        if self._extended:
            self._get_candidate_swap_list(node = node)
            node.edge_prob = prob
        else:
            super()._get_candidate_swap_list(node = node, pre_prob= True)
            node.edge_prob = [prob[idx]  for idx in node.candidate_swap_list]
        return node.value

    def _evaluate_for_training(self, node: MCTSNode):
        """

        """
        qubits, padding_mask, adj = self._get_data(node)
        qubits, padding_mask, adj = transform_batch(batch_data = (qubits, padding_mask, adj), device = torch.device("cpu") )
        #print("put in")
        self._input.put((self._id, qubits, padding_mask, adj))
        #print("put after")
        policy_scores, value_scores = self._output.recv()
        value = value_scores.numpy().squeeze()
        node.sim_value = value
        node.value = value 
        node.w = value
        prob = self._softmax(policy_scores).numpy()

        if self._extended:
            self._get_candidate_swap_list(node = node)
            node.edge_prob = prob
        else:
            super()._get_candidate_swap_list(node = node)
            node.edge_prob = [prob[idx]  for idx in node.candidate_swap_list]
        return node.value


    def _decide(self,node : MCTSNode)-> MCTSNode:
        """
        Decide which child node to move into 
        """
        node = self._get_best_child(node)
        node.parent = None
        #node.clear()
        return node

    def _expand(self, node: MCTSNode):
        """
        Open all child nodes of the current node by applying the swap gates in candidate swap list
        """
        if node is None:
            raise Exception("Node can't be None")

        for swap, prob in zip(node.candidate_swap_list, node.edge_prob):
            node.add_child_node_by_swap_gate(swap, prob)
   
    def _get_candidate_swap_list(self, node: MCTSNode):
        node.candidate_swap_list = [i  for i in range(self._coupling_graph.num_of_edge) ]

    def _get_best_child(self, node: MCTSNode)-> MCTSNode:
        """
        Get the child with highest score of the current node as the next root node. 
        """
        res_node = None
        score = -1
        idx = -1
        prob = np.ndarray(len(node.candidate_swap_list), dtype = np.float)
        value = np.ndarray(len(node.candidate_swap_list), dtype = np.float)
        for i, child in enumerate(node.children):
            prob[i] = child.visit_count
            value[i] = self._gamma * child.value + child.reward
        #print(2)
        prob = prob / np.sum(prob)
        node.visit_count_prob = prob
        if self._mode == MCTSMode.TRAIN:
            #print(1)
            if self._num_of_swap_gates < 30:
                idx = np.random.choice(prob.shape[0], p = prob)
            else: 
                idx = np.argmax(prob)
            #print(idx)
        elif self._mode == MCTSMode.SEARCH:
            idx = np.argmax(prob)
        elif self._mode == MCTSMode.EVALUATE:
            idx = np.argmax(value)
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
            UCB = self._upper_confidence_bound_with_predictor(node = child)
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

     
    def random_simulate(self, logical_circuit: Circuit, init_mapping: List[int]):    
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
      
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        
        
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 

        cur_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy(), parent =None)
        super()._get_candidate_swap_list(cur_node)
        num_of_swap_gates = 0
        while not cur_node.is_terminal_node():
            qubits, padding_mask, adj = self._get_data(cur_node)
            qubits, padding_mask, adj= transform_batch(batch_data = (qubits, padding_mask, adj), device = self._device)
            qubits, padding_mask, adj = qubits[None,:,:], padding_mask[None,:], adj[None, :, :]
            policy_scores, _ = self._model(qubits, padding_mask, adj)
            
            extended_prob = self._softmax(policy_scores.squeeze()).detach().to(torch.device('cpu')).numpy()
            #print(p)
            prob = np.array([  extended_prob[swap_index] 
                       for swap_index in cur_node.candidate_swap_list ])
            
            prob = prob / np.sum(prob)
            # prob = np.ones(len(cur_node.candidate_swap_list) , dtype = np.float32)
            
            # prob = prob / np.sum(prob)
           # print(prob)
            swap_idx = np.random.choice(len(prob), p = prob)
            #swap_idx = np.argmax(prob)
            cur_node.update_by_swap_gate(swap_index = cur_node.candidate_swap_list[swap_idx])
            super()._get_candidate_swap_list(cur_node)
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
            qubits, padding_mask, adj = self._get_data(cur_node)
            qubits, padding_mask, adj = transform_batch(batch_data = (qubits, padding_mask, adj), device = self._device)
            qubits, padding_mask, adj = qubits[None,:,:], padding_mask[None,:], adj[None,:]
            policy_scores, _ = self._model(qubits, padding_mask, adj)
            
            p = self._softmax(policy_scores.squeeze()).detach().to(torch.device('cpu')).numpy()
            #print(p)
            swap_idx = np.random.choice(p.shape[0], p = p)
            cur_node.update_by_swap_gate(swap_index = swap_idx)
            num_of_swap_gates += 1
            print(cur_node.num_of_gates)

        return num_of_swap_gates


        
        



