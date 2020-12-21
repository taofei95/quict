#from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable, Optional, Iterable, Union
import copy
from queue import deque
from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.gate.gate import *
from QuICT.core.exception import *

import numpy as np
import networkx as nx

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
    def __init__(self, circuit: Circuit = None):
        if circuit is not None:
            self._transform_from_QuICT_circuit(circuit = circuit)
        else:
            self._dag = nx.DiGraph()
        self._front_layer = None
    
    def __getitem__(self, index):
        """
        return the gate node corresponding to the index  in the DAG
        """
        return self._dag.nodes[index]

    @property
    def dag(self) -> nx.DiGraph:
        """
        """
        return self._dag
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

        """
        return self._dag.successors(vertex)

    def get_node_attributes(self, vertex: int):
        pass        

    def _transform_from_QuICT_circuit(self, circuit : Circuit):
        qubit_mask = np.zeros(circuit.circuit_length(), dtype = np.int32) -1
        num_gate = 0
        self._dag = nx.DiGraph()
        for gate in circuit.gates:
            self._dag.add_node(num_gate, gate = gate)
            if gate.controls + gate.targets == 1:
                self._add_edge_in_dag(num_gate, gate.targ, qubit_mask)
            elif gate.controls + gate.targets == 2:
                self._add_edge_in_dag(num_gate, gate.targ, qubit_mask)  
                self._add_edge_in_dag(num_gate, gate.carg, qubit_mask)
            else:
                raise Exception(str("The gate is not single qubit gate or two qubit gate"))
            num_gate = num_gate + 1
   
    def _add_edge_in_dag(self, cur_node: int, qubit: int, qubit_mask: np.ndarray):
        if qubit < len(qubit_mask):
            if qubit_mask[qubit] != -1:
                self._dag.add_edge(qubit_mask[qubit], cur_node)
            qubit_mask[qubit] = cur_node
        else:
            raise Exception(str("   "))

class CouplingGraph:
    def __init__(self, coupling_graph: List[Tuple] = None):
        if coupling_graph is not None:
            self._transform_from_list(coupling_graph = coupling_graph)
        else:
            self._coupling_graph = nx.Graph()

        self._size = self._coupling_graph.size()

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
        if self._coupling_graph[vertex_i, vertex_j] is not None:
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
    

class MCTSNode:
    #TODO: qubit_mask -> __init__.property
    def __init__(self, circuit_in_dag: DAG = None, coupling_graph: CouplingGraph = None, front_layer: List[int] = None, cur_mapping: List[int] = None, 
                 parent: MCTSNode =None, swap_with_edge: SwapGate = None):
        """
        Parameters
        ----------
        circuit_in_dag : 
        parent : 
        """
        #self._physical_circuit: Circuit  
        # TODO : circuit_in_dag -> remaining_circuit_dag 
        self._circuit_in_dag = circuit_in_dag

        # TODO : cur_mapping -> cur_mapping
        self._cur_mapping = cur_mapping
        self._coupling_graph = coupling_graph
        self._front_layer = front_layer
        # self._coupling_graph = coupling_graph
        self._parent = parent

        self._swap_with_edge = swap_with_edge
        #self.epsilon = epsilon
        self._children: List[MCTSNode] = [] 
         
        self._excution_list: List[BasicGate] = []  
        self._candidate_swap_list: List[SwapGate] = []
        self._qubit_mask: List[int] = []
        #self._front_layer = []
        self._visit_count = 0 
        self._NNC = None
        self._c = 0
       
        # TODO
        self._value = 0
        self._reward = 0
        
        # TODO
        self._v = 0
        self._q = 0

        if circuit_in_dag is not None and cur_mapping is not None and coupling_graph is not None:
            # self._update_qubit_mask()  
            # self._update_excution_list()
            # self._update_candidate_swap_list()
            self.update_node()

    
    @property
    def coupling_graph(self)->CouplingGraph:
        """
        """
        return self._coupling_graph

    @property
    def circuit_in_dag(self)->DAG:
        """
        """
        return self._circuit_in_dag
    @property
    def reward(self)->int:
        """
        """
        return self._reward
    
    @property
    def value(self)->float:
        """
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
    def c(self)->float:
        """
        value of the current node. v = argmax(q)
        """
        return self._c

    @property
    def front_layer(self)->List[int]:
        """

        """

        return self._front_layer

    @property 
    def excution_list(self)->List[BasicGate]:
        """
        """
        
        return self._excution_list

    @property
    def candidate_swap_list(self)->List[SwapGate]:
        """
        """

        return self._candidate_swap_list
    
    @property
    def cur_mapping(self)->List[int]:
        """
        the cur_mapping of physical qubits to logical qubits
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
        """
        self._visit_count = value
    @property
    def parent(self)->MCTSNode:
        """

        """
        return self._parent

    @property
    def children(self)->List[MCTSNode]:
        """

        """
        return self._children

    @property
    def upper_confidence_bound(self)->float:
        """
        The upper confidence bound of the node
        """
        return self.reward + self.value + self.c* np.sqrt(np.log2(self.parent.visit_count)/ self.visit_count)
    @property
    def swap_with_edge(self)->SwapGate:
        """
        The swap gate associated with the edge from the current node's parent to itself  
        """
        return self._swap_with_edge

    @swap_with_edge.setter
    def swap_with_edge(self, swap: SwapGate):
        self._swap_with_edge = swap
    
    def update_node(self):
        """
        update the node's property with the new cur_mapping or front_layer
        """
        self._update_qubit_mask()  
        self._update_excution_list()
        self._update_candidate_swap_list()

    def add_child_node(self, swap: SwapGate):
        """
        add a child node by applying the swap gate
        """
        if isinstance(swap, SwapGate):
            l_control = swap.carg
            l_target = swap.targ
            p_control = self.cur_mapping[l_control]
            p_target  = self.cur_mapping[l_target]
            if self._coupling_graph.is_adjacent(p_control, p_target):
                cur_mapping = self.cur_mapping.copy()
                cur_mapping[l_control] = p_target
                cur_mapping[l_target] = p_control
                child_node = MCTSNode(circuit_in_dag = self.circuit_in_dag, coupling_graph = self.coupling_graph, 
                             front_layer = self.front_layer.copy(), cur_mapping = cur_mapping, parent = self,
                             swap_with_edge = swap)
                self._children.append(child_node)
                
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")
        
    
    def is_leaf_node(self):
        """
        
        """
        if len(self.children):
            return False
        else:
            return True
            
    def is_terminal_node(self):
        """

        """
        if len(self.front_layer):
            return False
        else:
            return True
        
    
    def _update_candidate_swap_list(self):
        """
        all the swap gates associated with the front layer of the current node
        """
        qubits_set = self._get_invloved_qubits()
        candidate_swap_set = set()
        n = self.coupling_graph.size()

        for qubit in qubits_set:
            for adj in self.coupling_graph.get_adjacent_vertex():
                candidate_swap_set.add(frozenset(qubit,adj))
                
        candidate_swap_list = []
        for swap_index_set in candidate_swap_set:
            swap_index_list = list(swap_index_set)
            GateBuilder.setGateType(GATE_ID['Swap'])
            GateBuilder.setCargs(swap_index_list[0])
            GateBuilder.setTargs(swap_index_list[1])
            candidate_swap_list.append(GateBuilder.getGate())
        
        self._candidate_swap_list =  candidate_swap_list

    def _update_qubit_mask(self):
        """
        """
        self._qubit_mask = self._get_invloved_qubits()

    def _get_invloved_qubits(self)-> List[int] :
        """
        """
        qubit_set = set()
        for gate in self._front_layer:
            if self._gate_qubits(gate) == 1:
                qubit_set.add(self._gate_target(gate))
            else:
                qubit_set.add(self._gate_control(gate))
                qubit_set.add(self._gate_target(gate))

        return list(qubit_set)
    
    def _update_excution_list(self):
        """
        the gates that can be excuted immediately  with the qubit cur_mapping 
        and update the front layer and qubit mask of the nodes

        """
        self._excution_list = []
        fl_list = self._front_layer
        index = 0
        #reward = 0
        while index < len(fl_list):
            gate = fl_list[index]
            if self._gate_qubits(gate) == 1:
                self._excution_list.append(self._map_to_physics(gate_in_dag = gate))
                target = self._gate_target(gate)
                self._qubit_mask[target] = 0
                self._update_fl_list(fl_list = fl_list, gate_in_dag = gate)
                fl_list.remove(index)
            else:
                control = self._gate_control(gate)
                target = self._gate_target(gate)
                if self.coupling_graph.is_adjacent(control, target):
                    self._excution_list.append(self._circuit_in_dag[gate])
                    fl_list.remove(index)
                    self._qubit_mask[control] = 0
                    self._qubit_mask[control] = 0
                    self._update_fl_list(fl_list = fl_list, gate_in_dag = gate)
                else:
                    index = index + 1
        self._reward = len(self._excution_list)
        self._front_layer = fl_list

    def _update_fl_list(self, fl_list: List[int], gate_in_dag: int):
        """
        """
        for suc in self._circuit_in_dag.get_successor_nodes(gate_in_dag):
            if self._gate_qubits(suc) == 1:
                target = self._gate_target(suc)
                if self._qubit_mask[target] == 0:
                    fl_list.append(suc)
                    self._qubit_mask[target] = 1
            else:
                control = self._gate_control(suc)
                target = self._gate_target(suc)
                if self._qubit_mask[control] == 0 and self._qubit_mask[target] == 0:
                    fl_list.append(suc)
                    self._qubit_mask[control] = 1
                    self._qubit_mask[target] = 1


    def _map_to_physics(self, gate_in_dag: int)-> BasicGate:
        """
        """
        gate = self._circuit_in_dag[gate_in_dag]['gate'].copy()
        control = self._gate_control(gate_in_dag)
        target = self._gate_target(gate_in_dag)
        gate.cargs = control
        gate.cargs = target
        return gate

    def _gate_qubits(self, gate_in_dag: int)->int:
        """
        """
        return self._circuit_in_dag[gate_in_dag]['gate'].controls + self._circuit_in_dag[gate_in_dag]['gate'].controls

    def _gate_control(self, gate_in_dag: int)->int:
        """
        """
        return self._cur_mapping[self._circuit_in_dag[gate_in_dag]['gate'].carg]

    def _gate_target(self, gate_in_dag: int)->int:  
        """
        """
        return self._cur_mapping[self._circuit_in_dag[gate_in_dag]['gate'].targ]




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


