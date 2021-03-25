#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:51:46
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   experience_pool.py
from __future__ import division
import numpy as np
import torch

from QuICT.qcda.mapping._mcts_base import *

class ExperiencePool(object):
    @staticmethod
    def _fill_adj(array: np.ndarray):
        out = np.zeros(len(array)-1, dtype = array.dtype)
        for i in range(len(out)):
            if array[i+1] == -1:
                out[i] = array[0]
            else:
                out[i] = array[i+1]
        return out

    def __init__(self,coupling_graph: CouplingGraph, max_capacity: int = 100000):
        self._max_capacity = max_capacity
        if isinstance(coupling_graph, CouplingGraph):
            self._coupling_graph = coupling_graph
        elif isinstance(coupling_graph, list):
            self._coupling_graph = CouplingGraph(coupling_graph)
        else:
            raise Exception("The type is not supported")
        self._adj_list = []
        self._feature_list = []
        #self._front_layer_list = []
        # self._rewards_list = []
        # self._sim_value = []
        self._value_list = []
        self._action_probability_list = [] 
        self._candidate_action_list = self._coupling_graph.edges

        self._train_idx_list = []
        self._evaluate_idx_list = []

    def size(self)->int:
        """
        The number of samples in the pool
        """
        return len(self._adj_list)


    def train_set_size(self)->int:
        """
        """
        return len(self._train_idx_list)


    def evaluate_set_size(self)->int:
        """
        """
        return len(self._evaluate_idx_list)

    def feature_dim(self)->int:
        """
        The dimension of the feature vector 
        """
        if len(self._feature_list) == 0:
            return 0
        return self._feature_list[0].shape[1]

    def num_of_class(self)->int:
        """
        The number of distinct actions
        """
        return self._coupling_graph.num_of_edge

    def extend(self, state_list: List[np.ndarray], qubit_mapping_list: List[np.ndarray], action_probability_list: List[np.ndarray], value_list: List[int]):
        for i  in range(len(state_list)):
            self.push(state_list[i], qubit_mapping_list[i], action_probability_list[i], value_list[i])
    

    
    def push(self, state: np.ndarray, qubit_mapping: np.ndarray, action_probability: np.ndarray, value: int = 0): 
        adj = np.apply_along_axis(self._fill_adj, axis = 1, arr = np.concatenate([np.arange(0,state.shape[0])[:,np.newaxis], state[:,0:4]],axis = 1))
        #adj = state[:, 0:4]
        self._adj_list.append(adj)
        qubit_indices = qubit_mapping[state[:,4:5]]
        self._feature_list.append( self._coupling_graph.node_feature[qubit_indices,:].reshape(state.shape[0],-1))
    
        self._action_probability_list.append(action_probability)
        
        self._value_list.append(value)

        while len(self._adj_list) >= self._max_capacity:
            self.pop() # Maybe to pop random one instead of the first one

    def update_sim_val(self, index, sim_val):
        self._sim_val[index] = sim_val

    def pop(self, index = 0):
        self._state_list.pop(index)

    def split_data(self, quota: float):
        indices = np.random.permutation(range(len(self._adj_list)))
        split_point = int(quota * len(self._adj_list))
        self._evaluate_idx_list = indices[:split_point]
        self._train_idx_list = indices[split_point:]


    def clear(self):
        pass

    def load_data(self, file_path: str):
        self._adj_list = list(np.load(f"{file_path}/adj_list.npy", allow_pickle = True  ))
        self._feature_list = list(np.load(f"{file_path}/feature_list.npy", allow_pickle = True))
        self._value_list = list(np.load(f"{file_path}/value_list.npy", allow_pickle = True))
        self._action_probability_list = list(np.load(f"{file_path}/action_probability_list.npy", allow_pickle = True ))

    def save_data(self, file_path: str):
        np.save(f"{file_path}/adj_list.npy", self._adj_list)
        np.save(f"{file_path}/feature_list.npy", self._feature_list)
        np.save(f"{file_path}/value_list.npy", self._value_list)
        np.save(f"{file_path}/action_probability_list.npy", self._action_probability_list)

    def get_batch_data(self, batch_size: int = 32, device: torch.device = torch.device("cpu")):
        if self._train_idx_list.shape[0] == 0:
            self._train_idx_list = list(range(len(self._adj_list)))
        
        # print(batch_size)
        indices = np.random.choice(self._train_idx_list, batch_size)
        # print(batch_size)
        return self._get_chosen_data(indices, device)
   
    
    def get_evaluate_data(self, start: int, end: int, device: torch.device):
        
        indices = self._evaluate_idx_list[start:end]

        return self._get_chosen_data(indices, device)
        

    def _get_chosen_data(self, indices: np.ndarray, device: torch.device):
        adj_list = np.take(indices = indices, a = self._adj_list, axis = 0)
        adj_list = self._get_adj_list(adj_list)
        graph_pool = self._get_graph_pool(adj_list, device = device)
        adj = torch.from_numpy(np.concatenate(adj_list, axis = 0)).to(device).long()
        feature = torch.from_numpy(np.concatenate(np.take(indices = indices, a = self._feature_list, axis = 0 ), axis = 0)).to(device).float()
        value = torch.from_numpy(np.array(np.take(indices = indices, a = self._value_list))).to(device).float()
        action_probability = torch.from_numpy(np.vstack(np.take(indices = indices, a = self._action_probability_list))).to(device).float()
        
        return adj, graph_pool, feature, value, action_probability

    def _get_adj_list(self, adj_list: List[np.ndarray])->List[np.ndarray]:
        idx_bias = 0
        # print(adj_list)
        for i in range(len(adj_list)):
            adj_list[i] = adj_list[i] + idx_bias 
            idx_bias +=  adj_list[i].shape[0]
        
        #print(adj_list)
        return adj_list

    def _get_graph_pool(self, batch_graph: List[np.ndarray], device: torch.device):
        """
        """
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph))
        
        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            elem.extend([1] * len(graph))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0, 1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]])).to(device)

        return graph_pool

  


