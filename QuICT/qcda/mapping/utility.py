from __future__ import annotations
import copy
import os


from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable, Optional, Iterable, Union,Set
from enum import Enum
from pickle import Pickler

from dataclasses import dataclass


import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from exception import *
from dag import *
from coupling_graph import *

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.gate.gate import *
from QuICT.core.exception import *
from QuICT.core.layout import *


def is_two_qubit_gate_equal(s1: List[int], s2: List[int])->bool:
    """
    Indicate whether two two-qubit gates is same
    """
    if ( s1[0] == s2[0] and s1[1] == s2[1] ) or ( s1[0] == s2[1] and s1[1] == s2[0] ) :
        return True
    else:
        return False


def f(x: np.ndarray)->np.ndarray:
    return np.piecewise(x, [x<0, x==0, x>0], [0, 0.001, lambda x: x])



def transform_batch(batch_data: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device)->Tuple[torch.LongTensor, torch.FloatTensor]:
    qubits_list, padding_mask_list, adj_list = batch_data
    
    # padding_mask_list = np.concatenate([np.zeros(shape = (padding_mask_list.shape[0], 1), dtype = np.int32) , padding_mask_list], axis = 1)
    # qubits_list = np.concatenate([np.zeros(shape = (qubits_list.shape[0], 1 ,2), dtype = np.uint8) , qubits_list+2], axis = 1)
        
    padding_mask = torch.from_numpy(padding_mask_list).to(device).to(torch.uint8)
    qubits = torch.from_numpy(qubits_list).to(device).long()

    adj_list = get_adj_list(adj_list)
    if isinstance(adj_list, list):
        adj = torch.from_numpy(np.concatenate(adj_list, axis = 0)).to(device).long()
    else:
        adj =torch.from_numpy(adj_list).to(device).long()
    
    return qubits, padding_mask, adj

def get_adj_list(adj_list: List[np.ndarray])->np.ndarray:
    if isinstance(adj_list, list):
        idx_bias = 0
        # print(adj_list)
        for i in range(len(adj_list)):
            adj_list[i] = adj_list[i] + idx_bias 
            idx_bias +=  adj_list[i].shape[0]
        #print(adj_list)
    return adj_list

def get_graph_pool(batch_graph: List[np.ndarray], device: torch.device)-> torch.FloatTensor:
        """
        """
        if isinstance(batch_graph, list):
            num_of_graph = len(batch_graph)
            num_of_nodes = 0
            bias = [0]
            for i in range(num_of_graph):
                num_of_nodes += batch_graph[i].shape[0]
                bias.append(num_of_nodes)

            elem = torch.ones(num_of_nodes, dtype = torch.float)
            idx = torch.zeros([2, num_of_nodes ], dtype = torch.long)
            for i in range(num_of_graph):
                v = torch.arange(start = 0, end = batch_graph[i].shape[0], dtype = torch.long)
                idx[0, bias[i] : bias[i+1]] = i
                idx[1, bias[i] : bias[i+1]] = v
            graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([num_of_graph, num_of_nodes])).to(device)

        else:
            elem = torch.ones(batch_graph.shape[0], dtype = torch.float)
            idx = torch.zeros([2, batch_graph.shape[0] ], dtype = torch.long)
            idx[0,:] = 0
            idx[1,:] = torch.arange(start = 0, end = batch_graph.shape[0], dtype = torch.long)
            graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([1,batch_graph.shape[0]])).to(device)

        return graph_pool




class EdgeProb:
    def __init__(self, circuit: DAG = None, coupling_graph: CouplingGraph = None, qubit_mapping: List[int] = None, gates: List[int] = None , num_of_qubits: int = 0):
        self._circuit = circuit
        self._coupling_graph = coupling_graph
        self._qubit_mapping = qubit_mapping
        self._gates = gates
        self._inverse_qubit_mapping = [-1 for _ in range(max(len(qubit_mapping), num_of_qubits))]
        for i, elm in enumerate(qubit_mapping):
            self._inverse_qubit_mapping[elm] = i  


    def __call__(self, swap_index: int = -1)-> int:
       
        if swap_index == -1:
            return self._neareast_neighbour_count(self._gates, self._qubit_mapping)
        
        swap_gate = self._coupling_graph.get_swap_gate(swap_index)
        if isinstance(swap_gate, SwapGate) is not True:
             raise TypeException("swap gate","other gate")
        qubit_mapping = self._change_mapping_with_single_swap(swap_gate)
        
        return self._neareast_neighbour_count(self._gates, qubit_mapping)


    def _neareast_neighbour_count(self, gates: List[int], cur_mapping: List[int])-> int:
        """
        Caculate the sum of the distance of all the gates in the front layer on the physical device
        """
        NNC = 0
        for gate in gates:
            NNC = NNC + self._get_logical_gate_distance_in_device(cur_mapping, gate)
        return NNC

    def _get_logical_gate_distance_in_device(self, cur_mapping: List[int], gate: int)->int:
        """
        return the distance between the control qubit and target qubit of the given gate  on the physical device 
        """
        if self._circuit[gate]['gate'].type() == GATE_ID['Swap']:
            qubits = self._circuit[gate]['gate'].targs
        else:
            qubits = [ self._circuit[gate]['gate'].carg, self._circuit[gate]['gate'].targ ]
        
        return  self._coupling_graph.distance(cur_mapping[qubits[0]], cur_mapping[qubits[1]])


    def _change_mapping_with_single_swap(self, swap_gate: SwapGate)->List[int]:
        """
        Get the new mapping changed by a single swap, e.g., the mapping [0, 1, 2, 3, 4] of 5 qubits will be changed
        to the mapping [1, 0, 2, 3, 4] by the swap gate SWAP(0,1).
        """
        res_mapping = self._qubit_mapping.copy()
        if isinstance(swap_gate, SwapGate):
            p_target = swap_gate.targs
            l_target = [ self._inverse_qubit_mapping[i] for i in p_target ]
            if self._coupling_graph.is_adjacent(p_target[0], p_target[1]):
                if not l_target[0] == -1:
                    res_mapping[l_target[0]] = p_target[1]
                if not l_target[1] == -1:
                    res_mapping[l_target[1]] = p_target[0]
            else:
                raise MappingLayoutException()
        else:
                raise TypeException("swap gate","other gate")
        return res_mapping


# class GNNConfig(object):
#     def __init__(self, 
#                 num_of_gates: int = 150,
#                 maximum_capacity: int = 100000,
#                 num_of_nodes: int = 150,
#                 maximum_circuit: int = 5000,
#                 minimum_circuit: int = 50,
#                 batch_size: int = 10,
#                 ff_hidden_size: int = 128,
#                 num_self_att_layers: int =4,
#                 dropout: float = 0.5,
#                 num_U2GNN_layers: int = 1,
#                 value_head_size: int =256,
#                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#                 graph_name: str = 'ibmq20',
#                 learning_rate: float = 0.1,
#                 weight_decay: float = 1e-4,
#                 num_of_epochs: int = 50,
#                 num_of_process: int = 16,
#                 feature_update: bool = False,
#                 loss_c: float = 10,
#                 mcts_c: float = 20,
#                 gat: bool = False,
#                 d_embed: int = 64,
#                 d_model: int = 64,
#                 n_head: int = 4,
#                 n_layer: int = 4,
#                 n_encoder: int = 8,
#                 n_gat: int = 1,
#                 hid_dim: int = 128,
#                 dim_feedforward: int = 128):

#         self.num_of_gates = num_of_gates
#         self.maximum_capacity = maximum_capacity
#         self.num_of_nodes = num_of_nodes
#         self.maximum_circuit = maximum_circuit
#         self.minimum_circuit = minimum_circuit
#         self.batch_size = batch_size
#         self.ff_hidden_size = ff_hidden_size
#         self.num_self_att_layers = num_self_att_layers
#         self.dropout = dropout
#         self.num_U2GNN_layers = num_U2GNN_layers
#         self.value_head_size = value_head_size
#         self.device = device
#         self.graph_name = graph_name 
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.num_of_epochs = num_of_epochs
#         self.num_of_process  = num_of_process
#         self.feature_update = feature_update
#         self.loss_c = loss_c
#         self.mcts_c = mcts_c
#         self.gat = gat
#         self.d_embed = d_embed
#         self.d_model = d_model
#         self.n_head = n_head
#         self.n_layer = n_layer
#         self.n_encoder = n_encoder
#         self.n_gat = n_gat
#         self.hid_dim = hid_dim
#         self.dim_feedforward = dim_feedforward
@dataclass  
class GNNConfig(object):
    num_of_gates: int = 150
    maximum_capacity: int = 100000
    num_of_nodes: int = 150
    maximum_circuit: int = 5000
    minimum_circuit: int = 50
    gamma: int = 0.7
    selection_times: int = 40
    sim_method: int = 0
    batch_size: int = 10
    ff_hidden_size: int = 128
    num_self_att_layers: int =4
    dropout: float = 0.5
    num_U2GNN_layers: int = 1
    value_head_size: int = 256
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_name: str = 'ibmq20'
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    num_of_epochs: int = 50
    num_of_process: int = 16
    feature_update: bool = False
    loss_c: float = 10
    mcts_c: float = 20
    num_of_swap_gates: int = 15
    gat: bool = False
    d_embed: int = 64
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 4
    n_encoder: int = 8
    n_gat: int = 2
    hid_dim: int = 128
    dim_feedforward: int = 128



default_config = GNNConfig(maximum_capacity = 200000, num_of_gates = 150, maximum_circuit = 1500, minimum_circuit = 200, batch_size = 256,
                        ff_hidden_size = 128, num_self_att_layers=4, dropout = 0.5, value_head_size = 128, gamma = 0.7, 
                        num_U2GNN_layers=2, learning_rate = 0.001, weight_decay = 1e-4, num_of_epochs = 1000, device = torch.device( "cuda"),
                        graph_name = 'ibmq20',num_of_process = 10, feature_update = True, gat = False, n_gat = 2, mcts_c = 20, loss_c = 10,
                        num_of_swap_gates = 15, sim_method = 2 ) 

@dataclass
class SharedMemoryName:
    adj_name: str
    label_name: str
    value_name: str
    qubits_name: str
    num_name: str
    action_probability_name: str


class SimMode(Enum):
    """
    """
    AVERAGE = 0
    MAX = 1


class MCTSMode(Enum):
    """
    """
    TRAIN = 0
    EVALUATE = 1
    SEARCH = 2
    EXTENDED_PROB = 3



class RLMode(Enum):
    """
    """
    WARMUP = 0
    SELFPALY = 1 

class EvaluateMode(Enum):
    SEARCH = 0
    EXTENDED_PROB = 1
    PROB = 2


class Benchmark(Enum):
    REVLIB = 0
    RANDOM = 1

small_benchmark = ("rd84_142",
                   "adr4_197",
                   "radd_250",
                   "z4_268",
                   "sym6_145",
                   "misex1_241",
                   "rd73_252",
                   "cycle10_2_110",
                   "square_root_7",
                   "sqn_258",
                   "rd84_253") 


    