import numpy as np
#from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from QuICT.models import *

import numpy as np

class Dag:
    def __init__(self, circuit : Circuit):
        pass

    def get_front_layer(self):
        pass


class MonteCarloTreeNode:
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
    def __init__(self, **params):
        """
        initialize the Monte Carlo tree search with the given parameters 
        """
        
    def search(self, root_node:MonteCarloTreeNode):
        """
        """
        pass

    def expand(self):
        """
        open all child nodes of the  current node by applying the swap gates in candidate swap list
        """
        pass

    def rollout(self, method: str):
        """
        complete a heuristic search from the current node
        """
        pass

    def backpropagate(self, reward: float):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        pass

    
    def select(self):
        """
        select the child node with highest score
        """
        pass

    def eval(self, method: str):
        """
        evaluate the vlaue of the current node by DNN method
        """
        pass


class TableBasedMCTS(MCTSBase):
    def __init__(self):
        pass


class RLBasedMCTS(MCTSBase):
    def  __init__(self):
       self.experience_pool = []