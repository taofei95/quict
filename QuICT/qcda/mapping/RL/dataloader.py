from __future__ import division
import numpy as np
import torch

from multiprocessing import shared_memory, Lock, sharedctypes 
from QuICT.qcda.mapping.mcts_node import *
from utility import *



class DataLoader(object):

    def __init__(self, max_capacity: int = 1000000, num_of_nodes: int = 150, num_of_class: int = 43):
        
        self._num_of_nodes = num_of_nodes
        self._max_capacity = max_capacity
        self._num_of_class = num_of_class

        num_of_neighgour = 5
        self._idx = 0
    
        self._label_list = np.ndarray(shape = (max_capacity), dtype = np.int32)
        self._num_list = np.ndarray(shape = (max_capacity), dtype = np.int32)
        self._adj_list = np.ndarray(shape = (max_capacity, num_of_nodes, num_of_neighgour ), dtype = np.int32)
        self._qubits_list = np.ndarray(shape = (max_capacity, num_of_nodes, 2), dtype= np.int32)
        self._value_list = np.ndarray(shape = (max_capacity) , dtype = np.float)
        self._action_probability_list = np.ndarray(shape = (max_capacity, num_of_class) , dtype = np.float)


        self._train_idx_list = []
        self._evaluate_idx_list = []

    def size(self)->int:
        """
        The number of samples in the pool
        """
        return self._idx


    def train_set_size(self)->int:
        """
        """
        return len(self._train_idx_list)


    def evaluate_set_size(self)->int:
        """
        """
        return len(self._evaluate_idx_list)


    def num_of_class(self)->int:
        """adj_shm
        The number of distinct actions
        """
        return self._num_of_class

    def extend(self, adj: np.ndarray, qubits: np.ndarray, action_probability: np.ndarray, value: np.ndarray, circuit_size: np.ndarray, swap_label: np.ndarray, num: int = 20):
        if self._idx < self._max_capacity:
            end = min(self._idx + num, self._max_capacity)
            idx = self._idx
            input_end = end - self._idx
            self._num_list[idx : end] = circuit_size[0 : input_end]
            
            self._label_list[idx : end] =  swap_label[0 : input_end]

            self._adj_list[idx : end, :, :] = adj[0 : input_end, :, :]
            self._qubits_list[idx : end, :, :] = qubits[0 : input_end, :, :]
            self._action_probability_list[idx : end, :] = action_probability[0 : input_end, :] 
            self._value_list[idx : end] = value[0 : input_end]
            self._idx += input_end
            print(self._idx)
        else:
            raise Exception("the experience pool is fullfilled")
    
    def split_data(self, quota: float):
        print(self._idx)
        indices = np.random.permutation(range(self._idx))
        split_point = int(quota * self._idx)
        self._evaluate_idx_list = indices[:split_point]
        self._train_idx_list = indices[split_point:]


    def load_data(self, file_path: str):
        self._label_list[:] = np.load(f"{file_path}/label_list.npy", allow_pickle = True)
        self._num_list[:] = np.load(f"{file_path}/num_list.npy", allow_pickle = True)
        self._adj_list[:] = np.load(f"{file_path}/adj_list.npy", allow_pickle = True)
        self._qubits_list[:] = np.load(f"{file_path}/qubits_list.npy", allow_pickle = True)
        
       
        
        self._value_list[:] = np.load(f"{file_path}/value_list.npy", allow_pickle = True)
        self._action_probability_list[:] = np.load(f"{file_path}/action_probability_list.npy", allow_pickle = True )
        
        with open(f"{file_path}/metadata.txt",'r') as f:
            self._idx = int(f.readline())

    def save_data(self, file_path: str):
        np.save(f"{file_path}/label_list.npy", self._label_list)
        np.save(f"{file_path}/num_list.npy", self._num_list)
        np.save(f"{file_path}/adj_list.npy", self._adj_list)
        np.save(f"{file_path}/qubits_list.npy", self._qubits_list)
        np.save(f"{file_path}/value_list.npy", self._value_list)
        np.save(f"{file_path}/action_probability_list.npy", self._action_probability_list)
        with open(f"{file_path}/metadata.txt",'w') as f:
            f.write("%d"%(self._idx))

    def get_batch_data(self, batch_size: int = 32):
        if isinstance(self._train_idx_list, list):
            self._train_idx_list = np.arange(self._idx)
        
        # print(batch_size)
        indices = np.random.choice(self._train_idx_list, batch_size)
        #print(indices)
        # print(batch_size)
        return self._get_chosen_data(indices)
   
    def get_train_data(self, start: int, end: int):
        

        indices = self._train_idx_list[start:end]

        return self._get_chosen_data(indices)

    def get_evaluate_data(self, start: int, end: int):
        

        indices = self._evaluate_idx_list[start:end]

        return self._get_chosen_data(indices)

    def _get_chosen_data(self, indices: np.ndarray):
        
        qubits_list = np.take(indices = indices, a = self._qubits_list, axis = 0)
        label_list = np.take(indices = indices, a = self._label_list, axis = 0)
        num_list = np.take(indices = indices, a = self._num_list, axis = 0)
        adj_list = np.take(indices = indices, a = self._adj_list, axis = 0)
        value_list = np.take(indices = indices, a = self._value_list, axis = 0)
        action_probability_list = np.take(indices = indices, a = self._action_probability_list, axis = 0)
        padding_mask_list = np.zeros(shape = (qubits_list.shape[0], qubits_list.shape[1]), dtype = np.uint8)
        for i,idx in enumerate(num_list):
            padding_mask_list[i, idx:] = 1
            # adj_list_modified.append(adj_list[i,0:idx,:])
            # feature_list_modified.append(feature_list[i,0:idx,:])
            
        return  (qubits_list,
                padding_mask_list, 
                adj_list,  
                value_list, 
                action_probability_list, 
                label_list)

    
    