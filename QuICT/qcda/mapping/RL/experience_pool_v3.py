#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:51:46
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   experience_pool.py
from __future__ import division
import numpy as np
import torch

from multiprocessing import shared_memory, Lock, sharedctypes 
from QuICT.qcda.mapping._mcts_base import *
from utility import *



class ExperiencePool(object):

    def __init__(self, max_capacity: int = 100000, num_of_nodes: int = 50, num_of_class = 43, feature_dim: int = 20, shm_name: SharedMemoryName = None, idx_name: str = None, lock: Lock = None):
        
        self._idx_shm = shared_memory.SharedMemory(name = idx_name)
        self._max_capacity = max_capacity
        self._num_of_class = num_of_class

        self._adj_shm = shared_memory.SharedMemory(name = shm_name.adj_list_name)
        self._feature_shm = shared_memory.SharedMemory(name = shm_name.feature_list_name)
        self._action_probability_shm = shared_memory.SharedMemory(name = shm_name.action_probability_name)
        self._value_shm = shared_memory.SharedMemory(name = shm_name.value_list_name)
        
        self._idx = self._idx_shm.buf 

        self._adj_list = np.ndarray(shape = (max_capacity, num_of_nodes, 4 ) , dtype = np.int32, buffer = self._adj_shm.buf)
        self._feature_list = np.ndarray(shape = (max_capacity, num_of_nodes, feature_dim) , dtype = np.float, buffer = self._feature_shm.buf)
        self._value_list = np.ndarray(shape = (max_capacity) , dtype = np.float, buffer = self._value_shm.buf)
        self._action_probability_list = np.ndarray(shape = (max_capacity, num_of_class) , dtype = np.float, buffer = self._action_probability_shm.buf)


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

    def feature_dim(self)->int:
        """
        The dimension of the feature vector 
        """
        if len(self._feature_list) == 0:
            return 0
        return self._feature_list[0].shape[1]

    def num_of_class(self)->int:
        """adj_shm
        The number of distinct actions
        """
        return self._num_of_class

    def close(self):
        del self._adj_list
        del self._feature_list
        del self._value_list
        del self._action_probability_list 
        
        self._idx_shm.close()
        self._adj_shm.close()
        self._feature_shm.close()
        self._value_shm.close()
        self._action_probability_shm.close()

    def extend(self, adj: np.ndarray, feature: np.ndarray, action_probability: np.ndarray, value: np.ndarray, num: int = 20):
        if self._idx > self._max_capacity:
                raise Exception("The pool has been fullfilled")
        end = min(self._idx + num, self._max_capacity)
        input_end = end - self._idx
        lock.acquire()
        try:
            self._adj_list[self._idx : end, :, :] = adj[0 : input_end, :, :]
            self._feature_list[self._idx : end, :, :] = feature[0 : input_end, :, :]
            self._action_probability_list[self._idx : end, :] = action_probability[0 : input_end, :] 
            self._value_list[self._idx : end] = value[0 : input_end]
            self._idx += num
        finally:
            lock.release()

    

    
    def push(self, adj: np.ndarray, feature: np.ndarray, action_probability: np.ndarray, value: int = 0): 
        if self._idx > self._max_capacity:
                raise Exception("The pool has been fullfilled")
        lock.acquire()
        try:
            self._adj_list[self._idx, :, :] = adj
            self._feature_list[self._idx, :, :] = feature
            self._action_probability_list[self._idx, :] = action_probability 
            self._value_list[self._idx] = value
            self._idx += 1
        finally:
            lock.release()


    def split_data(self, quota: float):
        indices = np.random.permutation(range(self._idx))
        split_point = int(quota * self._idx)
        self._evaluate_idx_list = indices[:split_point]
        self._train_idx_list = indices[split_point:]


    def clear(self):
        pass

    def load_data(self, file_path: str):
        self._adj_list[:] = list(np.load(f"{file_path}/adj_list.npy", allow_pickle = True  ))
        self._feature_list[:] = list(np.load(f"{file_path}/feature_list.npy", allow_pickle = True))
        self._value_list[:] = list(np.load(f"{file_path}/value_list.npy", allow_pickle = True))
        self._action_probability_list[:] = list(np.load(f"{file_path}/action_probability_list.npy", allow_pickle = True ))
        

    def save_data(self, file_path: str):
        np.save(f"{file_path}/adj_list.npy", self._adj_list)
        np.save(f"{file_path}/feature_list.npy", self._feature_list)
        np.save(f"{file_path}/value_list.npy", self._value_list)
        np.save(f"{file_path}/action_probability_list.npy", self._action_probability_list)

    def get_batch_data(self, batch_size: int = 32):
        if self._train_idx_list.shape[0] == 0:
            self._train_idx_list = list(range(len(self._adj_list)))
        
        # print(batch_size)
        indices = np.random.choice(self._train_idx_list, batch_size)
        # print(batch_size)
        return self._get_chosen_data(indices)
   
    
    def get_evaluate_data(self, start: int, end: int):
        
        indices = self._evaluate_idx_list[start:end]

        return self._get_chosen_data(indices)

    def _get_chosen_data(self, indices: np.ndarray):
        adj_list = np.take(indices = indices, a = self._adj_list, axis = 0)
        feature_list = np.take(indices = indices, a = self._feature_list, axis = 0 )
        value_list = np.take(indices = indices, a = self._value_list)
        action_probability_list = np.take(indices = indices, a = self._action_probability_list)
        
        return adj_list,  feature_list, value_list, action_probability_list

    
    
        
  




    

  


