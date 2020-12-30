#from collections import defaultdict
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable, Optional, Iterable, Union,Set
import copy
from queue import deque
from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.gate.gate import *
from QuICT.core.exception import *

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class MappingLayoutException(Exception):
    """
    
    """
    def __init__(self):
        string = str("The control and target qubit of the gate is not adjacent on the physical device")
        Exception.__init__(self, string)

class LengthException(Exception):
    """

    """
    def __init__(self):
        string = str("The length of two objects does not match ")
        Exception.__init__(self, string)



class DAG:
    def __init__(self, circuit: Circuit = None,  mode = 1):
        """
        Params:
            circuit: the logical ciruit
            mode: deal with all the gates in the circuit when mode  = 1 
                    and deal with only the two-qubit gates when mode =2 
        """
        self._mode = mode
        if circuit is not None:
            if mode == 1:
                self._transform_from_circuit(circuit = circuit)
            elif mode == 2:
                self._construct_two_qubit_gates_circuit(circuit = circuit)
            else:
                raise Exception("The mode is not supported")
        else:
            self._dag = nx.DiGraph()
        self._front_layer = None
    
    def __getitem__(self, index):
        """
        return the gate node corresponding to the index  in the DAG
        """
        return self._dag.nodes[index]
    @property
    def size(self) -> int:
        """
        the number of nodes in DAG
        """
        return len(self._dag.nodes)
    @property
    def dag(self) -> nx.DiGraph:
        """
        the directed acyclic graph representation of the circuit
        """
        return self._dag

    @property
    def qubit_mask(self)->np.ndarray:
        """
        the qubit mask stores the index of current closet gate in the DAG, e.g.,
        given the  circuit with the sequential two-qubit gates {(0,1),(1,2),(2,3),(3,4),(4,1)},
        the qubit mask should be (0,0,1,2,3), which means the first and second physical 
        should be allocated to the first gate for it's the first gate on the qubit wires and so on.  
        """
        if self._mode == 1:
            return None
        elif self._mode == 2:
            return self._initial_qubit_mask
        else:
            raise Exception("The mode is not supported")


    @property
    def front_layer(self)-> List[int]:
        """
        The front layer of the DAG, which is the list of all the nodes with zero in-degree
        """
        if self._front_layer is not None:
            return self._front_layer
        else:
            front_layer = []
            for v in list(self._dag.nodes()):
                if self._dag.in_degree[v] == 0:
                    front_layer.append(v)
            self._front_layer = front_layer
            return self._front_layer
  
    def get_successor_nodes(self, vertex: int)->Iterable[int]:
        """
        get the succeeding nodes of the current nodes
        """
        return self._dag.successors(vertex)

        
    def get_predecessor_nodes(self, vertex: int)->Iterable[int]:
        """
        get the preceeding nodes of the current nodes
        """
        return self._dag.predecessors(vertex)

    def get_egde_qubit(self, vertex_i: int, vertex_j: int):
        """
        the qubit associated with the edge from vertex i to vertex j
        """
        return self._dag.edges[vertex_i, vertex_j]['qubit']



    def _transform_from_circuit(self, circuit : Circuit):
        """
        transform the whole circuit into a directed acyclic graph
        """
        self._num_gate = 0
        self._qubit_mask = np.zeros(circuit.circuit_length(), dtype = np.int32) -1
        self._initial_qubit_mask = np.zeros(circuit.circuit_length(), dtype = np.int32) -1
        self._depth = np.zeros(circuit.circuit_size(), dtype = np.int32) 
        
        self._dag = nx.DiGraph()
        for gate in circuit.gates:
            self._dag.add_node(self._num_gate, gate = gate, depth = self._gate_depth(gate))
            if gate.controls + gate.targets == 1:
                self._add_edge_in_dag(gate.targ)
            elif gate.controls + gate.targets == 2:
                if gate.type() == GATE_ID['Swap']:
                    self._add_edge_in_dag(gate.targs[0])  
                    self._add_edge_in_dag(gate.targs[1])
                else:
                    self._add_edge_in_dag(gate.targ)  
                    self._add_edge_in_dag(gate.carg)
            else:
                raise Exception(str("The gate is not single qubit gate or two qubit gate"))
            self._num_gate = self._num_gate + 1

    def _construct_two_qubit_gates_circuit(self, circuit: Circuit):
        """
        transform the sub circuit only with two-qubit gates  in the original circuit into a directed acyclic graph
        """
        self._num_gate = 0
        self._qubit_mask = np.zeros(circuit.circuit_length(), dtype = np.int32) -1
        self._initial_qubit_mask = np.zeros(circuit.circuit_length(), dtype = np.int32) -1
        self._depth = np.zeros(circuit.circuit_size(), dtype = np.int32)
        self._dag = nx.DiGraph()
        for gate in circuit.gates:
            if gate.is_single(): 
                pass
            elif gate.controls + gate.targets == 2:
                if self._is_duplicate_gate(gate) is not True:
                    self._dag.add_node(self._num_gate, gate = gate, depth = self._gate_depth(gate))
                    if gate.type() == GATE_ID['Swap']:
                        self._add_edge_in_dag(gate.targs[0])  
                        self._add_edge_in_dag(gate.targs[1])
                    else:
                        self._add_edge_in_dag(gate.targ)  
                        self._add_edge_in_dag(gate.carg)
            else:
                raise Exception(str("The gate is not single qubit gate or two qubit gate"))
            
            self._num_gate = self._num_gate + 1
    
    def _is_duplicate_gate(self, gate: BasicGate)->bool:
        """
        indicate wether the gate share the same qubits with its preceeding gate
        """
        if gate.type() == GATE_ID['Swap']:
            qubits = (gate.targs[0], gate.targs[1])
        else:
            qubits = (gate.carg, gate.targ)
       
        if self._qubit_mask[qubits[0]] != -1 and self._qubit_mask[qubits[0]] == self._qubit_mask[qubits[1]]:
            return True
        else:
            return False

    def _gate_depth(self, gate: BasicGate)->int:
        """

        """
        if gate.controls + gate.targets == 1:
            self._depth[self._num_gate] = self._gate_before_qubit_depth(gate.targ)

        elif gate.controls + gate.targets == 2:
            self._depth[self._num_gate] = max(self._gate_before_qubit_depth(gate.targ), self._gate_before_qubit_depth(gate.carg))
        else:
            raise Exception(str("The gate is not single qubit gate or two qubit gate"))
        
        return self._depth[self._num_gate]
    
    def _gate_before_qubit_depth(self, qubit: int)->int:
        """
        """
        if self._qubit_mask[qubit] == -1:
            return 0
        else:
            return self._depth[self._qubit_mask[qubit]] + 1

    def _add_edge_in_dag(self, qubit: int):
        """
        """
        if qubit < len(self._qubit_mask):
            if self._qubit_mask[qubit] != -1:
                self._dag.add_edge(self._qubit_mask[qubit], self._num_gate, qubit = qubit )
            else:
                 self._initial_qubit_mask[qubit] = self._num_gate
            self._qubit_mask[qubit] = self._num_gate
        else:
            raise Exception(str("   "))
    
    def draw(self):
        """
        draw the DAG of the circuit with 
        """
        plt.figure(figsize = (10,10))
        nx.draw(G = self._dag, pos = nx.multipartite_layout(self._dag, subset_key = 'depth'), node_size = 50, width= 1, arrowsize = 2, font_size =12, with_labels = True)
        #nx.draw(G = self._dag)
        plt.savefig("dag.png")
        plt.close()


class CouplingGraph:
    def __init__(self, coupling_graph: List[Tuple] = None):
        if coupling_graph is not None:
            self._transform_from_list(coupling_graph = coupling_graph)
        else:
            self._coupling_graph = nx.Graph()

        self._size = self._coupling_graph.size()
        self._cal_shortest_path()
    @property
    def size(self):
        """
        the number of vertex (physical qubits) in the coupling graph
        """
        return self._size  

    @property
    def coupling_graph(self):
        """
        """
        return self._coupling_graph

    def is_adjacent(self, vertex_i: int, vertex_j: int) -> bool:
        """
        indicate wthether the two vertices are adjacent on the coupling graph
        """
        if (vertex_i, vertex_j) in self._coupling_graph.edges:
            return True
        else:
            return False


    def get_adjacent_vertex(self, vertex: int) ->Iterable[int]:
        """
        return the adjacent vertices of  the given vertex
        """
        return self._coupling_graph.neighbors(vertex)

    def distance(self, vertex_i: int, vertex_j: int)->int:
        """
        the distance of two vertex on the coupling graph of the physical devices
        """
        return self._shortest_paths[vertex_i][vertex_j]


    def _transform_from_list(self,  coupling_graph : List[Tuple]):
        """
        construct the coupling graph from the list of tuples
        """
        res_graph = nx.Graph(coupling_graph)
        self._coupling_graph = res_graph

    def _cal_shortest_path(self):
        """
        calculate the shortest path between every two vertices on the graph
        """
        self._shortest_paths = nx.algorithms.shortest_paths.dense.floyd_warshall(G = self._coupling_graph)
    def draw(self):
        """
        """ 
      
        nx.draw(G = self._coupling_graph)
        plt.savefig("coupling_graph.png")
        plt.close()

class MCTSNode:
    def __init__(self, circuit_dag: DAG = None, coupling_graph: CouplingGraph = None, front_layer: List[int] = None, cur_mapping: List[int] = None, 
                qubit_mask: np.ndarray = None, parent: MCTSNode = None, swap_of_edge: SwapGate = None):
        """
        Parameters
        ----------
            circuit_dag: the directed acyclic graph representation of the circuit, which is stored 
                         in a static memory and shared by all the MCTS nodes.
            coupling_graph: the physical device's graph.
            front_layer: the set of all the nodes in the DAG with zero in-degree. 
            cur_mapping: the current mapping of logical qubit to physical qubits 
            qubit_mask: the qubit mask stores the index of current closet gate in the DAG, e.g.,
                      given the  circuit with the sequential two-qubit gates {(0,1),(1,2),(2,3),(3,4),(4,1)},
                      the qubit mask should be (0,0,1,2,3), which means the first and second physical 
                      should be allocated to the first gate for it's the first gate on the qubit wires and so on.  
            parent: the parent node of the current node.
            swap_of_edge: the swap gate associated with the edge from the node's parent to itself.
        """ 

        self._circuit_dag = circuit_dag
        self._gate_index = []
        self._cur_mapping = cur_mapping
        self._coupling_graph = coupling_graph
        self._front_layer = front_layer
        self._qubit_mask = qubit_mask
        self._parent = parent

        self._swap_of_edge = swap_of_edge


        self._children: List[MCTSNode] = [] 
         
        self._excution_list: List[int] = []  
        
        self._candidate_swap_list: List[SwapGate] = []
    
        self._visit_count = 1 
       
        self._value = 0
        self._reward = 0
        
        self._v = 0
        self._q = 0

        if circuit_dag is not None and cur_mapping is not None and coupling_graph is not None:
            self.update_node()

    
    @property
    def coupling_graph(self)->CouplingGraph:
        """
        the physical device's graph
        """
        return self._coupling_graph

    @property
    def circuit_dag(self)->DAG:
        """
        the directed acyclic graph representation of the circuit
        """
        return self._circuit_dag
    @property
    def reward(self)->int:
        """
        the reward of the swap gate which transforms the mapping of node's parent to its. It equals the 
        number of the excutable gates under the current mapping.
        """
        return self._reward
    
    @property
    def value(self)->float:
        """
        the long-term value of the current node, the definition refers to the paper.
        """
        return self._value
    
    @value.setter
    def value(self, value: float):
        """
        """
        self._value = value


    @property
    def q(self)->float:
        """
        score of the current node
        """
        return self._q
    

    @property
    def v(self)->float:
        """
        value of the current node. v = argmax(q)
        """
        return self._v

    @property
    def qubit_mask(self)->np.ndarray:
        """
        the qubit mask stores the index of current closet gate in the DAG, e.g.,
                      given the  circuit with the sequential two-qubit gates {(0,1),(1,2),(2,3),(3,4),(4,1)},
                      the qubit mask should be (0,0,1,2,3), which means the first and second physical 
                      should be allocated to the first gate for it's the first gate on the qubit wires and so on.  
            parent: the parent node of the current node.
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
        the set of all the nodes in the DAG with zero in-degree. 

        """

        return self._front_layer

    @property 
    def excution_list(self)->List[BasicGate]:
        """
        the list of the excutable gates under the current mapping
        """
        
        return self._excution_list

    @property
    def candidate_swap_list(self)->List[SwapGate]:
        """
        the list of the candidate swap gate, which has at least one qubit dominated by the gates in the front layer  
        """

        return self._candidate_swap_list
    
    @property
    def cur_mapping(self)->List[int]:
        """
        the cur_mapping of logical qubits to physical qubits
        """
        return self._cur_mapping

    @cur_mapping.setter
    def cur_mapping(self, cur_mapping: List[int]):
        """
        assign the new mapping to  'cur_mapping' of the node  
        """
        self._cur_mapping = cur_mapping


    @property
    def visit_count(self)->int:
        """
        the number of  times the node has been visted
        """
        return self._visit_count
    @visit_count.setter
    def visit_count(self, value:int):
        """
        the times of the node been visited by the search algorithm
        """
        self._visit_count = value
    @property
    def parent(self)->MCTSNode:
        """
        the parent node of the current node.
        """
        return self._parent

    @property
    def children(self)->List[MCTSNode]:
        """
        the set of the child nodes of the current node
        """
        return self._children

    @property
    def upper_confidence_bound(self)->float:
        """
        The upper confidence bound of the node
        """
        return self.reward + self.value + self.c* np.sqrt(np.log2(self.parent.visit_count) / self.visit_count)
    @property
    def swap_of_edge(self)->SwapGate:
        """
        The swap gate associated with the edge from the current node's parent to itself  
        """
        return self._swap_of_edge

    @swap_of_edge.setter
    def swap_of_edge(self, swap: SwapGate):
        self._swap_of_edge = swap
    
    def update_node(self):
        """
        update the node's property with the new cur_mapping or front_layer
        """
        self._update_excution_list()
        self._update_candidate_swap_list()

    def add_child_node(self, swap: SwapGate):
        """
        add a child node by applying the swap gate
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
                             swap_of_edge = swap.copy())
                self._children.append(child_node)
                
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")
        
    
    def is_leaf_node(self):
        """
        indicate whether the node is a leaf node of the current the monte carlo tree
        """
        if len(self.children) != 0:
            return False
        else:
            return True
            
    def is_terminal_node(self):
        """
        indicate whether the node is a terminal node, i.e., the whole logical circuit has been transformed into 
        the hardware-compliant circuit.
        """
        if len(self.front_layer)!=0:
            return False
        else:
            return True
        
    
    def _update_candidate_swap_list(self):
        """
        all the swap gates associated with the front layer of the current node
        """
        qubits_set = self._get_invloved_qubits()
        candidate_swap_set = set()
        n = self.coupling_graph.size

        for qubit in qubits_set:
            for adj in self.coupling_graph.get_adjacent_vertex(qubit):
                candidate_swap_set.add(frozenset([qubit,adj]))
                
        candidate_swap_list = []
        for swap_index_set in candidate_swap_set:
            swap_index_list = list(swap_index_set)
            GateBuilder.setGateType(GATE_ID['Swap'])
            GateBuilder.setTargs(swap_index_list)
            candidate_swap_list.append(GateBuilder.getGate())
        
        self._candidate_swap_list =  candidate_swap_list
   

    def _get_invloved_qubits(self)-> List[int] :
        """
        get the list of the physical qubits dominated by the gates in front layer 
        """
        qubit_set = set()
        for gate in self._front_layer:
            if self._gate_qubits(gate) == 1:
                qubit_set.add(self._gate_target(gate))
            elif self._is_swap(gate) is not True:
                qubit_set.add(self._gate_control(gate))
                qubit_set.add(self._gate_target(gate))
            else:
                qubit_set.add(self._gate_target(gate,0))
                qubit_set.add(self._gate_target(gate,1))

        return list(qubit_set)
    

    def _update_excution_list(self):
        """
        the gates that can be excuted immediately  with the qubit cur_mapping 
        and update the front layer and qubit mask of the nodes
        """

        self._excution_list = []
        fl_stack = deque(self._front_layer)
        self._front_layer = []
     
        while len(fl_stack) > 0:
           
            gate = fl_stack.pop()
               
            if self._is_swap(gate) is not True:
                control = self._gate_control(gate)
                target = self._gate_target(gate)
            else:
                control = self._gate_target(gate,0)
                target = self._gate_target(gate,1)
                
            if self.coupling_graph.is_adjacent(control, target):
                self._excution_list.append(gate)
                self._update_fl_list(stack = fl_stack, gate_in_dag = gate)
            else:
                self._front_layer.append(gate)
        self._reward = len(self._excution_list)

    
    def _update_fl_list(self, stack: deque, gate_in_dag: int):
        """
        update the front layer list when a gate is removed from the list
        """
        for suc in self.circuit_dag.get_successor_nodes(gate_in_dag):
            self._qubit_mask[self._edge_qubit(gate_in_dag, suc)] = suc
           
            if self._is_swap(suc) is not True:
                control = self._gate_control(suc)
                target = self._gate_target(suc)
            else:
                control = self._gate_target(suc,0)
                target = self._gate_target(suc,1)

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
        the qubit associated with the edge from the gate i to gate j in the DAG, i.e.,
        the qubit is shared by the consecutive two gates. 
        """
        return self.cur_mapping[self.circuit_dag.get_egde_qubit(vertex_i, vertex_j)]
       


    def _get_physical_gate(self, gate_in_dag: int)-> BasicGate:
        """
        get the physical gate of the given logical gate 
        """
        gate = self.circuit_dag[gate_in_dag]['gate'].copy()
        if self._gate_qubits(gate_in_dag) == 1:
            target = self._gate_target(gate_in_dag)
            gate.targs = target
        elif self._is_swap(gate_in_dag) is not True:
            control = self._gate_control(gate_in_dag)
            target = self._gate_target(gate_in_dag)
            gate.cargs = control
            gate.targs = target
        else:
            target_0 = self._gate_target(gate_in_dag,0)
            target_1 = self._gate_target(gate_in_dag,1)
            gate.targs = [target_0, target_1]
        return gate

    def _gate_qubits(self, gate_in_dag: int)->int:
        """
        the number of the qubits dominated by the gate.
        """
                   
        return self.circuit_dag[gate_in_dag]['gate'].controls + self.circuit_dag[gate_in_dag]['gate'].targets
    
    def _is_swap(self, gate_in_dag: int)->bool:
        """
        indicate whether the gate is a swap gate
        """
        return self.circuit_dag[gate_in_dag]['gate'].type() == GATE_ID['Swap']

    def _gate_control(self, gate_in_dag: int, index: int = 0)->int:
        """
        the physical qubit of the gate 'index'-th control qubit under the current mapping 
        """
        if  index < self.circuit_dag[gate_in_dag]['gate'].controls:
            return self._cur_mapping[self.circuit_dag[gate_in_dag]['gate'].cargs[index]]
        else:
            raise  IndexLimitException(self.circuit_dag[gate_in_dag]['gate'].controls, index)

    def _gate_target(self, gate_in_dag: int, index: int = 0)->int:  
        """
        the physical qubit of the gate 'index'-th target qubit under the current mapping 
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
        evaluate the vlaue of the current node by DNN method
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


