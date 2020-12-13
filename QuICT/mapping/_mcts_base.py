import numpy as np
#from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from QuICT.models import *

import numpy as np
import networkx as nx
class Dag:
    instances_count = 0
    def __init__(self, circuit : Circuit):
        if circuit is not None:
            self.dag = _transform_from_circuit(circuit = circuit)
        else 
            self.dag = nx.DiGraph()
    
    def get_front_layer(self):
        pass

    def read_from_qasm(self, file_name :str):
        pass
    def _tranform_from_circuit(self, circuit : Circuit):
        pass


class MCTSNode:
    def __init__(self, circuit: Dag, layout: List[int], parent=None):
        """
        Parameters
        ----------
        circuit : 
        parent : 
        """
        self.circuit = circuit
        self.layout = layout
        self.parent = parent
        #self.epsilon = epsilon
        self.children = []
        self.front_layer = []
        self.excution_list = []
        self.candidate_swap_list = []
        
        

        self.v = 0
        self.q = 0
    
    @property
    def q(self):
        """
        score of the current node
        """
        pass


    @property
    def v(self):
        """
        value of the current node. v = argmax(q)
        """
        pass


    @property
    def circuit(self):
        """
        logic circuit corresponding to the current node
        """
        pass



    def _add_child_node(self, swap: BasicGate):
        """
        add a child node by applying the swap gate
        """
        pass
    

    def is_leaf_node(self):
        """
        
        """
        pass

    def _best_child(self, **params):
        """

        """
        pass


    def _get_candidate_swap_list(self):
        """
        all the swap gates associated with the front layer of the current node
        """
        pass 
    
    def _get_excution_list(self):
        """
        the gates that can be excuted immediately  with the qubit layout
        """
        pass

    def _update_node(self):
        """
        remove the gates in excution list and update the circuit dag of the node
        """
        pass



class MCTSBase:
    @abstractmethod
    def __init__(self, coupling_graph:List[Tuple] = None,**params):
        """
        initialize the Monte Carlo tree search with the given parameters 
        """
        self.coupling_graph = coupling_graph
        self.shortest_path_coupling_graph,  self.shortest_path_coupling_graph =  _cal_shortest_path(self.coupling_graph)


    @abstractmethod
    def search(self, root_node : MCTSNode):
        """
        """
        pass

    @abstractmethod
    def _expand(self):
        """
        open all child nodes of the  current node by applying the swap gates in candidate swap list
        """
        pass

    @abstractmethod
    def _rollout(self, method: str):
        """
        complete a heuristic search from the current node
        """
        pass
    
    @abstractmethod
    def _backpropagate(self, reward: float):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        pass

    @abstractmethod
    def _select(self):
        """
        select the child node with highest score
        """
        pass
    @abstractmethod
    def _eval(self):
        """
        evaluate the vlaue of the current node by DNN method
        """
        pass

    def _cal_shortest_path(self.coupling_graph):
        """
        """
        pass

    def _read_coupling_graph(self, file_name: str):
        """
        """
        pass


