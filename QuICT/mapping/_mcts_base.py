#from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from copy import copy, deepcopy
from queue import deque
from QuICT.models import *
from QuICT.models._gate import *
from QuICT.exception import *

import numpy as np
import networkx as nx

class MappingLayoutException(Exception):
    """

    """
    def __init__(self):
        string = str("The control and target qubit of the gate is not adjacent on the physical device")
        Exception.__init__(self, string)

class DAG:
    def __init__(self, circuit: Circuit = None):
        if circuit is not None:
            self._dag = self._transform_from_circuit(circuit = circuit)
        else:
            self._dag = nx.DiGraph()
        self._front_layer = None
    @property
    def dag(self) -> nx.DiGraph:
        """
        """
        return self._dag
    @property
    def front_layer(self)-> List[int]:
        """
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

  
    def get_successor_nodes(self, vertex: int):
        return self._dag.successors(vertex)

    def get_node_attributes(self, vertex: int):
        pass

    def read_from_qasm(self, file_name :str):
        pass

    def _transform_from_circuit(self, circuit : Circuit):
        return nx.DiGraph()

class CouplingGraph:
    def __init__(self, coupling_graph: List[Tuple] = None):
        if coupling_graph is not None:
            self._coupling_graph = self._transform_from_list(coupling_graph = coupling_graph)
        else:
            self._coupling_graph = nx.Graph()
        self._shortest_path = self._cal_shortest_path()
        self._size = self._coupling_graph.size()

    @property
    def size(self):
        """
        """
        return self._size  
    @property
    def coupling_graph(self):
        """
        """
        return self._coupling_graph

    @property
    def shortest_path(self):
        """
        """
        return self._shortest_path
    
    @property
    def is_adjacent(self, vertex_i: int, vertex_j: int) -> bool:
        """

        """
        if self._coupling_graph[vertex_i, vertex_j] is not None:
            return True
        else:
            return False
    @property
    def get_adjacent_vertex(self, vertex: int) ->List[int]:
        """

        """
        pass 

    def _transform_from_list(self,  coupling_graph : List[Tuple]) -> nx.Graph:
        """
        """
        res_graph = nx.Graph()

        return res_graph

    def _cal_shortest_path(self):
        """
        """
        res = Tuple[List[int]]
        return res
    

class MCTSNode:

    def __init__(self, circuit: DAG = None, coupling_graph: CouplingGraph = None ,front_layer: List[int] = None, layout: List[int] = None, 
                 parent: MCTSNode =None):
        """
        Parameters
        ----------
        circuit : 
        parent : 
        """
        #self._physical_circuit: Circuit  
        self._circuit = circuit
        self._layout = layout
        self._coupling_graph = coupling_graph
        # self._coupling_graph = coupling_graph
        self._parent = parent
        #self.epsilon = epsilon
        self._children: List[Tuple[MCTSNode, SwapGate]] = [] 
        #self._front_layer = []
        if circuit is not None and layout is not None and coupling_graph is not None:
            self._excution_list = []
            self._front_layer = []
            self._excution_list, self._front_layer = self._get_excution_list()
            self._candidate_swap_list = self._get_candidate_swap_list()
        else:
            self._excution_list = []
            self._front_layer = []
            self._candidate_swap_list = []

        self._visit_count = 0 
        self._NNC = None
        self._c = 0
        self._value = 0
        self._reward = 0
        self._v = 0
        self._q = 0
    @property
    def coupling_graph(self):
        """
        """
        return self._coupling_graph

    @property
    def circuit(self):
        """
        """
        return self._circuit
    @property
    def reward(self):
        """
        """
        return self._reward
    
    @property
    def value(self):
        """
        """
        return self._value

    @property
    def q(self):
        """
        score of the current node
        """
        return self._q
    

    @property
    def v(self):
        """
        value of the current node. v = argmax(q)
        """
        return self._v
    @property
    def c(self):
        """
        value of the current node. v = argmax(q)
        """
        return self._c

    @property
    def front_layer(self):
        """

        """
        return self._front_layer

    @property 
    def excution_list(self):
        """
        """
        return self._excution_list

    @property
    def candidate_swap_list(self):
        """
        """
        return self._candidate_swap_list

    # @property
    # def logical_circuit(self):
    #     """
    #     logic circuit corresponding to the current node
    #     """
    #     return self._logical_circuit

    # @property
    # def physical_circuit(self):
    #     """
    #     """
    #     return self._physical_circuit

    # @property
    # def coupling_graph(self):
    #     """
    #     """
    #     return self._coupling_graph
    
    @property
    def layout(self):
        """
        the layout of physical qubits to logical qubits
        """
        return self._layout

    @layout.setter
    def layout(self, layout: List[int]):
        """
        """
        self._layout = layout


    @property
    def visit_count(self):
        """
        the number of  times the node has been visted
        """
        return self._visit_count
    @property
    def parent(self):
        """

        """
        return self._parent

    @property
    def children(self):
        """

        """
        return self._children

    @property
    def upper_confidence_bound(self):
        return self.q + self.v + self.c* np.sqrt(np.log2(self.parent.visit_count)/ self.visit_count)
    
    def add_child_node(self, swap: SwapGate):
        """
        add a child node by applying the swap gate
        """
        if isinstance(swap, SwapGate):
            l_control = swap.carg
            l_target = swap.targ
            p_control = self.layout[l_control]
            p_target  = self.layout[l_target]
            if self.coupling_graph.is_adjacent(p_control, p_target):
                layout = self.layout
                layout[l_control] = p_target
                layout[l_target] = p_control
                child_node = MCTSNode(circuit = self.circuit, front_layer = self.front_layer, layout = layout, parent = self)
                self._children.append(child_node)
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")
        
    
    def neareast_neighbour_count(self)-> int:
        """
        """
        if self._NNC is None:
            res: int = 0

            self._NNC = res
            return self._NNC
        else:
            return self._NNC


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
        

    def select_next_child(self):
        """
        return the child_node with highest UCB score
        """
        score = -1 
        node = MCTSNode()
        for child in self._children:
            if child[0].upper_confidence_bound > score:
                node = child[0] 
                score = node.upper_confidence_bound
        return  node  
    
    def decide_best_child(self):
        """
        return the child_node with highest v value and its corresponding swap operation
        """
        node = MCTSNode()
        res_swap =  BasicGate()
        score = -1
        for child in self._children:
            if child[0].v >score:
                node = child[0]
                score = node.v
                res_swap = child[1]

        return node, res_swap


    def _get_candidate_swap_list(self):
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
            GateBuilder.setGateType(GateType.Swap)
            GateBuilder.setCargs(swap_index_list[0])
            GateBuilder.setTargs(swap_index_list[1])
            candidate_swap_list.append(GateBuilder.getGate())
        
        return candidate_swap_list

    def _get_invloved_qubits(self):
        """
        """
        qubit_set = set()
        qubit_list = []
        for gate in self._front_layer:
            pass

        return qubit_list
    
    def _get_excution_list(self)-> List[BasicGate]:
        """
        the gates that can be excuted immediately  with the qubit layout
        """
        
        excution_list = []
        fl_queue = deque(iterable=self._front_layer)
        
        while True:
            add_list = []
            delete_list = []
            while len(fl_queue)  != 0:
                gate = fl_queue[0]
                control = self._circuit[gate]['gate'].carg
                target = self._circuit[gate]['gate'].targ
                if self.coupling_graph.is_adjacent(control, target):
                    excution_list.append(self._circuit[gate])
                    fl_queue.pop()
                    for suc in self._circuit.get_successor_nodes(gate):
                        pass
                                  
            if len(excution_list) == 0:
                break
        return excution_list
        

    def update_node(self):
        """
        remove the gates in excution list and update the circuit dag of the node
        """
        pass





class MCTSBase:

    def __init__(self, coupling_graph:List[Tuple] = None,**params):
        """
        initialize the Monte Carlo tree search with the given parameters 
        """
        self._coupling_graph = coupling_graph

    @property
    def coupling_graph(self):
        """
        """
        return self._coupling_graph

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


