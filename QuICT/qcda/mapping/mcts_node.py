#from collections import defaultdict
from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable, Optional, Iterable, Union,Set
from enum import Enum
from collections import deque

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from exception import *
from dag import *
from coupling_graph import *
from utility import *

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.gate.gate import *
from QuICT.core.exception import *
from QuICT.core.layout import *



class MCTSNode:

    def __init__(self, circuit_dag: DAG = None, coupling_graph: CouplingGraph = None, front_layer: List[int] = None, cur_mapping: List[int] = None, 
                qubit_mask: np.ndarray = None, parent: MCTSNode = None, swap_of_edge: SwapGate = None, prob_of_edge: float = 0):
        """
        Parameters
        ----------
            circuit_dag: The directed acyclic graph representation of the circuit, which is stored 
                         in a static memory and shared by all the MCTS nodes.
            coupling_graph: The physical device's graph.
            front_layer: The set of all the nodes in the DAG with zero in-degree. 
            cur_mapping: The current mapping of logical qubit to physical qubits 
            qubit_mask: The qubit mask stores the index of current closet gate in the DAG, e.g.,
                      given the  circuit with the sequential two-qubit gates {(0,1),(1,2),(2,3),(3,4),(4,1)},
                      the qubit mask should be (0,0,1,2,3), which means the first and second physical 
                      should be allocated to the first gate for it's the first gate on the qubit wires and so on.  
            parent: The parent node of the current node.
            swap_of_edge: The swap gate associated with the edge from the node's parent to itself.
        """ 

        self._circuit_dag = circuit_dag
        self._gate_index = []
        self._cur_mapping = cur_mapping
        self._coupling_graph = coupling_graph
        self._front_layer = front_layer
        self._qubit_mask = qubit_mask
        self._parent = parent
        self._swap_of_edge = swap_of_edge
        self._prob_of_edge = prob_of_edge
        self._children: List[MCTSNode] = []     
        self._execution_list: List[int] = []     
        self._candidate_swap_list: List[SwapGate] = []
        self._visit_count = 0   
        self._value = 0
        self._edge_prob: np.ndarray = None
        self._sim_value = 0
        self._reward = 0
        self._v = 0
        self._q = 0
        self._w = 0
        self._state = None
        
        if self._parent is None:
            self._num_of_gates = self._circuit_dag.size
        else:
            self._num_of_gates = self.parent.num_of_gates
        if circuit_dag is not None and cur_mapping is not None and coupling_graph is not None:
            self.update_node()
    

    @property
    def state(self)->Tuple[np.ndarray, np.ndarray]:
        return self._state

    @state.setter
    def state(self, s: Tuple[np.ndarray, np.ndarray]):
        self._state = s

    @property
    def num_of_gates(self):
        """
        The number of gates remaining in the circuit
        """
        return self._num_of_gates
    
    @num_of_gates.setter
    def num_of_gates(self, value):
        """
        """
        self._num_of_gates = value

    @property
    def coupling_graph(self)->CouplingGraph:
        """
        The physical device's graph
        """
        return self._coupling_graph

    @property
    def circuit_dag(self)->DAG:
        """
        The directed acyclic graph representation of the circuit
        """
        return self._circuit_dag
    @property
    def reward(self)->int:
        """
        The reward of the swap gate which transforms the mapping of node's parent to its. It equals the 
        number of the executable gates under the current mapping.
        """
        return self._reward
    


    @property
    def sim_value(self)->float:
        """

        """
        return self._sim_value
    
    @sim_value.setter
    def sim_value(self, value: float):
        """
        """
        self._sim_value = value

    @property
    def value(self)->float:
        """
        The long-term value of the current node, the definition refers to the paper.
        """
        # if self._num_of_gates == 0: 
        # #     raise Exception("number of gates is zero")
        #     return 1
        # if  self._visit_count == 0:
        #     return 1 - self._coupling_graph.size
        # else:
        #     self._value = 1 - self._sim_value / (self._visit_count *self._num_of_gates)
        
        return self._value
    
    @value.setter
    def value(self, value: float):
        """
        """
        self._value = value
    @property
    def prob_of_edge(self)->float:
        """
        The prior probability of the edge from the parent node to the current node
        """
        return self._prob_of_edge

    @property
    def edge_prob(self)-> np.ndarray:
        """
        The prior probability over the edges from the current node
        """
        return self._edge_prob
    
    @edge_prob.setter
    def edge_prob(self, p: np.ndarray):
        self._edge_prob = p

    @property
    def extended_edge_prob(self)-> np.ndarray:
        """
        """
        return self._extended_edge_prob

    @property
    def q(self)->float:
        """
        Score of the current node
        """
        return self._q

    @q.setter
    def q(self, value: float)->float:
        """
        """
        self._q = value
    
    @property
    def w(self)->float:
        """
        """
        return self._w
    
    @w.setter
    def w(self, value: float):
        """
        """
        self._w = value
    
    @property
    def v(self)->float:
        """
        Value of the current node. v = argmax(q)
        """
        return self._v

    @property
    def qubit_mask(self)->np.ndarray:
        """
        The qubit mask stores the index of current closet gate in the DAG, e.g.,
                      given the  circuit with the sequential two-qubit gates {(0,1),(1,2),(2,3),(3,4),(4,1)},
                      the qubit mask should be (0,0,1,2,3), which means the first and second physical 
                      should be allocated to the first gate for it's the first gate on the qubit wires and so on.  
        """
        return self._qubit_mask

    @qubit_mask.setter
    def qubit_mask(self, value: np.ndarray):
        """
        """
        self._qubit_mask = value
    @property
    def front_layer(self)->List[int]:
        """
        The set of all the nodes in the DAG with zero in-degree. 
        """
        return self._front_layer

    @property 
    def execution_list(self)->List[BasicGate]:
        """
        The list of the executable gates under the current mapping
        """
        
        return self._execution_list

    @property
    def candidate_swap_list(self)->List[SwapGate]:
        """
        The list of the candidate swap gate, which has at least one qubit dominated by the gates in the front layer  
        """
        return self._candidate_swap_list
    
    @property
    def cur_mapping(self)->List[int]:
        """
        The cur_mapping of logical qubits to physical qubits
        """
        return self._cur_mapping

    @cur_mapping.setter
    def cur_mapping(self, cur_mapping: List[int]):
        """
        Assign the new mapping to  'cur_mapping' of the node  
        """
        self._cur_mapping = cur_mapping

    @property
    def visit_count(self)->int:
        """
        The number of  times the node has been visited
        """
        return self._visit_count
   
    @visit_count.setter
    def visit_count(self, value:int):
        """
        The times of the node been visited by the search algorithm
        """
        self._visit_count = value
    
    @property
    def parent(self)->MCTSNode:
        """
        The parent node of the current node.
        """
        return self._parent
    @parent.setter
    def parent(self, p: MCTSNode):
        """
        """
        self._parent = p
    @property
    def children(self)->List[MCTSNode]:
        """
        The set of the child nodes of the current node
        """
        return self._children
   
    @property
    def swap_of_edge(self)->SwapGate:
        """
        The swap gate associated with the edge from the current node's parent to itself  
        """
        return self._swap_of_edge

    @swap_of_edge.setter
    def swap_of_edge(self, swap: SwapGate):
        self._swap_of_edge = swap
    
    def clear(self):
        self._parent = None
        self._children: List[MCTSNode] = []      
        self._visit_count = 0   
        self._value = 0
        self._sim_value = 0
        self._v = 0
        self._q = 0
        self._w = 0
      
    def update_node(self):
        """
        Update the node's property with the new cur_mapping or front_layer
        """
        self._update_execution_list()
        self._update_candidate_swap_list()

    def update_by_swap_gate(self, swap: SwapGate):
        """
        """
        if isinstance(swap, SwapGate):
            p_target = swap.targs
            l_target = [self.cur_mapping.index(p_target[i], 0, len(self.cur_mapping)) 
                        for i in  range(len(p_target)) ]
            if self._coupling_graph.is_adjacent(p_target[0], p_target[1]):

                temp = self._qubit_mask[p_target[0]]
                self._qubit_mask[p_target[0]] = self._qubit_mask[p_target[1]]
                self._qubit_mask[p_target[1]] = temp   
                     
                self._cur_mapping[l_target[0]] = p_target[1]
                self._cur_mapping[l_target[1]] = p_target[0]
                self.update_node()          
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")

    
    def add_child_node_by_swap_gate(self, swap: SwapGate, prob: float):
        """
        Add a child node by applying the swap gate
        """
        if isinstance(swap, SwapGate):
            p_target = swap.targs
            l_target = [self.cur_mapping.index(p_target[i], 0, len(self.cur_mapping)) 
                        for i in  range(len(p_target)) ]
            if self._coupling_graph.is_adjacent(p_target[0], p_target[1]):
                cur_mapping = self.cur_mapping.copy()
                qubit_mask = self.qubit_mask.copy()    

                temp = qubit_mask[p_target[0]]
                qubit_mask[p_target[0]] = qubit_mask[p_target[1]]
                qubit_mask[p_target[1]] = temp   
                     
                cur_mapping[l_target[0]] = p_target[1]
                cur_mapping[l_target[1]] = p_target[0]

                child_node = MCTSNode(circuit_dag = self.circuit_dag, coupling_graph = self.coupling_graph, 
                             front_layer = self.front_layer.copy(), qubit_mask = qubit_mask, cur_mapping = cur_mapping, parent = self,
                             swap_of_edge = swap.copy(), prob_of_edge = prob)
                self._children.append(child_node)           
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")
        
    
    def is_leaf_node(self):
        """
        Indicate whether the node is a leaf node of the current the monte carlo tree
        """
        if len(self.children) != 0:
            return False
        else:
            return True
            
    def is_terminal_node(self):
        """
        Indicate whether the node is a terminal node, i.e., the whole logical circuit has been transformed into 
        the hardware-compliant circuit.
        """
        if len(self.front_layer)!=0:
            return False
        else:
            return True
        
    def _update_candidate_swap_list(self):
        """
        All the swap gates associated with the front layer of the current node
        """
        qubits_set = self._get_invloved_qubits()
        candidate_swap_set = set()

        for qubit in qubits_set:
            for adj in self.coupling_graph.get_adjacent_vertex(qubit):
                candidate_swap_set.add(frozenset([qubit,adj]))
                
        candidate_swap_list = []
        for swap_index_set in candidate_swap_set:
            swap_index_list = list(swap_index_set)
            if  self.parent is None or self.parent.swap_of_edge is None or not is_two_qubit_gate_equal(swap_index_list, self.parent.swap_of_edge.targs):
                GateBuilder.setGateType(GATE_ID['Swap']) 
                GateBuilder.setTargs(swap_index_list)
                candidate_swap_list.append(GateBuilder.getGate())
        
        self._candidate_swap_list =  candidate_swap_list
        edge_prob_cal = EdgeProb(circuit = self._circuit_dag, coupling_graph = self._coupling_graph, qubit_mapping = self._cur_mapping, gates = self._front_layer)
        base =  edge_prob_cal()
        edge_prob = f(np.array([base - edge_prob_cal(swap) for swap in self._candidate_swap_list ],dtype=np.float))
        edge_prob = edge_prob / np.sum(edge_prob)
        self._edge_prob = edge_prob
        self._extended_edge_prob = self._extend_action_probability(action_probability = edge_prob, swap_edges = candidate_swap_list)


    def _extend_action_probability(self, action_probability: np.ndarray, swap_edges: List[SwapGate]):
        res = np.zeros(self._coupling_graph.num_of_edge, dtype = float)
        for p, swap in zip(action_probability, swap_edges):
            idx = self._coupling_graph.edge_label(swap)
            res[idx] = p
        return res


    def _get_invloved_qubits(self)-> List[int]:
        """
        Get the list of the physical qubits dominated by the gates in front layer 
        """
        qubit_set = set()
        for gate in self._front_layer:
            if self._gate_qubits(gate) == 1:
                qubit_set.add(self._get_gate_target(gate))
            elif self._is_swap(gate) is not True:
                qubit_set.add(self._get_gate_control(gate))
                qubit_set.add(self._get_gate_target(gate))
            else:
                qubit_set.add(self._get_gate_target(gate,0))
                qubit_set.add(self._get_gate_target(gate,1))
        return list(qubit_set)
    

    def _update_execution_list(self):
        """
        The gates that can be executed immediately  with the qubit cur_mapping 
        and update the front layer and qubit mask of the nodes
        """
        self._execution_list = []
        fl_stack = deque(self._front_layer)
        self._front_layer = []    
        while len(fl_stack) > 0:  
            gate = fl_stack.pop()       
            if self._is_swap(gate) is not True:
                control = self._get_gate_control(gate)
                target = self._get_gate_target(gate)
            else:
                control = self._get_gate_target(gate,0)
                target = self._get_gate_target(gate,1)   
                     
            if self.coupling_graph.is_adjacent(control, target):
                self._execution_list.append(gate)
                self._update_fl_stack(stack = fl_stack, gate_in_dag = gate)
            else:
                self._front_layer.append(gate)
        self._reward = len(self._execution_list)
        self._value = self._reward
        self._num_of_gates =self._num_of_gates - len(self._execution_list)
    
    def _update_fl_stack(self, stack: deque, gate_in_dag: int):
        """
        Update the front layer list when a gate is removed from the list
        """
        for suc in self.circuit_dag.get_successor_nodes(gate_in_dag):
            self._qubit_mask[self._edge_qubit(gate_in_dag, suc)] = suc      
            if self._is_swap(suc) is not True:
                control = self._get_gate_control(suc)
                target = self._get_gate_target(suc)
            else:
                control = self._get_gate_target(suc,0)
                target = self._get_gate_target(suc,1)
            if self._is_qubit_free(control, suc)  and  self._is_qubit_free(target, suc):
                stack.append(suc)
                self._qubit_mask[control] = suc
                self._qubit_mask[target] = suc

    def _is_qubit_free(self, qubit: int, gate: int)-> bool:
        """
        The qubit is free for the gate if and only if the qubit is not occupied by the other gate(qubit_mask[qubit] = -1)
        or the qubit has been allocated to the gate itself
        """
        return self._qubit_mask[qubit] == -1 or self._qubit_mask[qubit] == gate

    def _edge_qubit(self, vertex_i: int, vertex_j: int )->int:
        """
        The qubit associated with the edge from the gate i to gate j in the DAG, i.e.,
        the qubit is shared by the consecutive two gates. 
        """
        return self.cur_mapping[self.circuit_dag.get_egde_qubit(vertex_i, vertex_j)]
       
    def _get_physical_gate(self, gate_in_dag: int)-> BasicGate:
        """
        Get the physical gate of the given logical gate 
        """
        gate = self.circuit_dag[gate_in_dag]['gate'].copy()
        if self._gate_qubits(gate_in_dag) == 1:
            target = self._get_gate_target(gate_in_dag)
            gate.targs = target
        elif self._is_swap(gate_in_dag) is not True:
            control = self._get_gate_control(gate_in_dag)
            target = self._get_gate_target(gate_in_dag)
            gate.cargs = control
            gate.targs = target
        else:
            target_0 = self._get_gate_target(gate_in_dag,0)
            target_1 = self._get_gate_target(gate_in_dag,1)
            gate.targs = [target_0, target_1]
        return gate

    def _gate_qubits(self, gate_in_dag: int)->int:
        """
        The number of the qubits dominated by the gate.
        """            
        return self.circuit_dag[gate_in_dag]['gate'].controls + self.circuit_dag[gate_in_dag]['gate'].targets
    
    def _is_swap(self, gate_in_dag: int)->bool:
        """
        Indicate whether the gate is a swap gate
        """
        return self.circuit_dag[gate_in_dag]['gate'].type() == GATE_ID['Swap']

    def _get_gate_control(self, gate_in_dag: int, index: int = 0)->int:
        """
        The physical qubit of the gate 'index'-th control qubit under the current mapping 
        """
        if  index < self.circuit_dag[gate_in_dag]['gate'].controls:
            return self._cur_mapping[self.circuit_dag[gate_in_dag]['gate'].cargs[index]]
        else:
            raise  IndexLimitException(self.circuit_dag[gate_in_dag]['gate'].controls, index)

    def _get_gate_target(self, gate_in_dag: int, index: int = 0)->int:  
        """
        The physical qubit of the gate 'index'-th target qubit under the current mapping 
        """
        return self._cur_mapping[self.circuit_dag[gate_in_dag]['gate'].targs[index]]

    
    def copy(self)->MCTSNode:
        """
        """
        return MCTSNode(circuit_dag=self._circuit_dag, coupling_graph= self._coupling_graph,
                       front_layer=self._front_layer.copy(), qubit_mask=self._qubit_mask.copy(),cur_mapping=self._cur_mapping.copy())




class MCTSBase:

    def __init__(self, **params):
        """
        initialize the Monte Carlo tree search with the given parameters 
        """

    @property
    def coupling_graph(self):
        """
        """
        pass

    @abstractmethod
    def search(self, circuit : Circuit):
        """
        """
        pass

    @abstractmethod
    def _expand(self, cur_node : MCTSNode):
        """
        open all child nodes of the  current node by applying the swap gates in candidate swap list
        """
        pass

    @abstractmethod
    def _rollout(self, cur_node : MCTSNode, method: str):
        """
        complete a heuristic search from the current node
        """
        pass
    
    @abstractmethod
    def _backpropagate(self, cur_node : MCTSNode):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        pass

    @abstractmethod
    def _select(self, cur_node : MCTSNode):
        """
        select the child node with highest score
        """
        pass
    @abstractmethod
    def _eval(self, cur_node : MCTSNode):
        """
        evaluate the value of the current node by DNN method
        """
        pass

    @abstractmethod 
    def _decide(self,cur_node : MCTSNode):
        """

        """
        pass

    def _read_coupling_graph(self, file_name: str):
        """
        """
        pass


