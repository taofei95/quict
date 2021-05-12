#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:51:46
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   experience_pool.py
from __future__ import division

from multiprocessing import shared_memory, Lock, sharedctypes

from .utility import *


class ExperiencePool(object):

    def __init__(self, max_capacity: int = 1000000, num_of_nodes: int = 150, graph_name: str = None,
                 num_of_class: int = 43, num_of_swap_gates: int = 15):

        self._num_of_nodes = num_of_nodes
        self._max_capacity = max_capacity
        self._num_of_class = num_of_class
        self._num_of_swap_gates = num_of_swap_gates
        num_of_neighgour = 5
        self._lock = Lock()
        self._idx = sharedctypes.Value('i', 0, lock=False)
        self._label_shm = shared_memory.SharedMemory(create=True, size=max_capacity * 4)
        self._num_shm = shared_memory.SharedMemory(create=True, size=max_capacity * 4)
        self._adj_shm = shared_memory.SharedMemory(create=True, size=max_capacity * num_of_nodes * num_of_neighgour * 4)
        self._qubits_shm = shared_memory.SharedMemory(create=True, size=max_capacity * num_of_nodes * 2 * 4)
        self._action_probability_shm = shared_memory.SharedMemory(create=True, size=max_capacity * num_of_class * 8)
        self._value_shm = shared_memory.SharedMemory(create=True, size=max_capacity * 8)
        self._executed_gates_shm = shared_memory.SharedMemory(create=True, size=max_capacity * num_of_swap_gates * 4)

        self._label_list = np.ndarray(shape=(max_capacity), dtype=np.int32, buffer=self._label_shm.buf)
        self._num_list = np.ndarray(shape=(max_capacity), dtype=np.int32, buffer=self._num_shm.buf)
        self._adj_list = np.ndarray(shape=(max_capacity, num_of_nodes, num_of_neighgour), dtype=np.int32,
                                    buffer=self._adj_shm.buf)
        self._qubits_list = np.ndarray(shape=(max_capacity, num_of_nodes, 2), dtype=np.int32,
                                       buffer=self._qubits_shm.buf)
        self._value_list = np.ndarray(shape=(max_capacity), dtype=np.float, buffer=self._value_shm.buf)
        self._action_probability_list = np.ndarray(shape=(max_capacity, num_of_class), dtype=np.float,
                                                   buffer=self._action_probability_shm.buf)
        self._executed_gates_list = np.ndarray(shape=(max_capacity, num_of_swap_gates), dtype=np.int32,
                                               buffer=self._executed_gates_shm.buf)

        self._train_idx_list = []
        self._evaluate_idx_list = []

    def size(self) -> int:
        """
        The number of samples in the pool
        """
        return self._idx

    def train_set_size(self) -> int:
        """
        """
        return len(self._train_idx_list)

    def evaluate_set_size(self) -> int:
        """
        """
        return len(self._evaluate_idx_list)

    def feature_dim(self) -> int:
        """
        The dimension of the feature vector 
        """

        return self._feature_dim

    def num_of_class(self) -> int:
        """adj_shm
        The number of distinct actions
        """
        return self._num_of_class

    def close(self):
        del self._label_list
        del self._num_list
        del self._adj_list
        del self._qubits_list
        del self._value_list
        del self._action_probability_list
        del self._executed_gates_list

        self._label_shm.close()
        self._num_shm.close
        self._adj_shm.close()
        self._qubits_shm.close()
        self._value_shm.close()
        self._action_probability_shm.close()
        self._executed_gates_shm.close()

    def unlink(self):
        self._label_shm.unlink()
        self._num_shm.unlink()
        self._adj_shm.unlink()
        self._qubits_shm.unlink()
        self._value_shm.unlink()
        self._action_probability_shm.unlink()
        self._executed_gates_shm.unlink()

    def extend(self, adj: np.ndarray, qubits: np.ndarray, action_probability: np.ndarray, value: np.ndarray,
               circuit_size: np.ndarray, swap_label: np.ndarray, num: int = 20, executed_gates: np.ndarray = None):
        self._lock.acquire()
        try:
            if self._idx.value < self._max_capacity:
                end = min(self._idx.value + num, self._max_capacity)
                idx = self._idx.value
                input_end = end - self._idx.value
                self._num_list[idx: end] = circuit_size[0: input_end]

                self._label_list[idx: end] = swap_label[0: input_end]

                self._adj_list[idx: end, :, :] = adj[0: input_end, :, :]
                self._qubits_list[idx: end, :, :] = qubits[0: input_end, :, :]
                self._action_probability_list[idx: end, :] = action_probability[0: input_end, :]
                self._value_list[idx: end] = value[0: input_end]
                if executed_gates is not None:
                    self._executed_gates_list[idx: end, :] = executed_gates[0:input_end, :]
                self._idx.value += input_end
                print(self._idx.value)
            else:
                raise Exception("the experience pool is fullfilled")
        finally:
            self._lock.release()

    def push(self, adj: np.ndarray, qubits: np.ndarray, action_probability: np.ndarray, value: float = 0,
             circuit_size: int = 0, swap_label: int = 0, executed_gates: np.ndarray = None):
        # print(self._idx.value)
        self._lock.acquire()
        if self._idx.value >= self._max_capacity:
            raise Exception("The pool has been fullfilled")
        try:
            idx = self._idx.value
            self._label_list[idx] = swap_label
            self._num_list[idx] = circuit_size
            self._adj_list[idx, :, :] = adj
            self._qubits_list[idx, :, :] = qubits
            self._action_probability_list[idx, :] = action_probability
            self._value_list[idx] = value
            if executed_gates is not None:
                self._executed_gates_list[idx, :] = executed_gates
            self._idx.value += 1
        finally:
            self._lock.release()

    def split_data(self, quota: float):
        print(self._idx.value)
        indices = np.random.permutation(range(self._idx.value))
        split_point = int(quota * self._idx.value)
        self._evaluate_idx_list = indices[:split_point]
        self._train_idx_list = indices[split_point:]

    def clear(self):
        self._lock.acquire()
        try:
            if self._idx.value > 0:
                self._idx.value = 0
        finally:
            self._lock.release()

    def load_data(self, file_path: str):
        self._label_list[:] = np.load(f"{file_path}/label_list.npy", allow_pickle=True)
        self._num_list[:] = np.load(f"{file_path}/num_list.npy", allow_pickle=True)
        self._adj_list[:] = np.load(f"{file_path}/adj_list.npy", allow_pickle=True)
        self._qubits_list[:] = np.load(f"{file_path}/qubits_list.npy", allow_pickle=True)

        self._value_list[:] = np.load(f"{file_path}/value_list.npy", allow_pickle=True)
        self._action_probability_list[:] = np.load(f"{file_path}/action_probability_list.npy", allow_pickle=True)

        with open(f"{file_path}/metadata.txt", 'r') as f:
            self._idx.value = int(f.readline())

    def save_data(self, file_path: str):
        np.save(f"{file_path}/label_list.npy", self._label_list)
        np.save(f"{file_path}/num_list.npy", self._num_list)
        np.save(f"{file_path}/adj_list.npy", self._adj_list)
        np.save(f"{file_path}/qubits_list.npy", self._qubits_list)
        np.save(f"{file_path}/value_list.npy", self._value_list)
        np.save(f"{file_path}/action_probability_list.npy", self._action_probability_list)
        np.save(f"{file_path}/executed_gates_list.npy", self._executed_gates_list)
        with open(f"{file_path}/metadata.txt", 'w') as f:
            f.write("%d" % (self._idx.value))

    def get_batch_data(self, batch_size: int = 32):
        if self._train_idx_list.shape[0] == 0:
            self._train_idx_list = np.arange(self._idx.value)

        # print(batch_size)
        indices = np.random.choice(self._train_idx_list, batch_size)
        # print(indices)
        # print(batch_size)
        return self._get_chosen_data(indices)

    def get_train_data(self, start: int, end: int):

        indices = self._train_idx_list[start:end]

        return self._get_chosen_data(indices)

    def get_evaluate_data(self, start: int, end: int):

        indices = self._evaluate_idx_list[start:end]

        return self._get_chosen_data(indices)

    def _get_chosen_data(self, indices: np.ndarray):

        qubits_list = np.take(indices=indices, a=self._qubits_list, axis=0)
        label_list = np.take(indices=indices, a=self._label_list, axis=0)
        num_list = np.take(indices=indices, a=self._num_list, axis=0)
        adj_list = np.take(indices=indices, a=self._adj_list, axis=0)
        value_list = np.take(indices=indices, a=self._value_list, axis=0)
        action_probability_list = np.take(indices=indices, a=self._action_probability_list, axis=0)
        padding_mask_list = np.zeros(shape=(qubits_list.shape[0], qubits_list.shape[1]), dtype=np.bool)

        for i, idx in enumerate(num_list):
            padding_mask_list[i, idx:] = 1

        return (qubits_list,
                padding_mask_list,
                adj_list,
                value_list,
                action_probability_list,
                label_list)
