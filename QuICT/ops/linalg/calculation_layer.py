#!/usr/bin/env python
# -*- coding:utf8 -*-
import contextvars
import weakref
import cupy as cp
import numpy as np
import sys
from contextlib import contextmanager


from . import gpu_calculator as GPUCalculator


class CalculationLayer:
    def __init__(self, gpu_device: int = 0, memory_limit: int = None):
        if gpu_device != cp.cuda.runtime.getDevice() and gpu_device < cp.cuda.runtime.getDeviceCount():
            dev = cp.cuda.Device(gpu_device)
            cp.cuda.Device.use(dev)

        self.mempool = cp.get_default_memory_pool()
        self.pinned_mempool = cp.get_default_pinned_memory_pool()

        if memory_limit:
            self.mempool.set_limit(memory_limit*1024**3)     # memory_limit GiB

        # Set Memory allocator
        cp.cuda.set_allocator(self.mempool.malloc)

        # Memory Record
        self.mempool_used = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_memory()

    def clear_memory(self): 
        self.mempool_used.clear()
        self.mempool.free_all_blocks()
        self.pinned_mempool.free_all_blocks()

    def htod(self, target):
        """ mv target from host into GPU device. """
        if type(target) is not cp.ndarray:
            # mmr_ptr = cp.cuda.memory.alloc(target.nbytes)
            cp_t = cp.array(target)
            weak_r = weakref.ref(cp_t)
            self.mempool_used.append(cp_t)

            return weak_r

        raise(f"The given value has been added in the GPU.")

    def dtoh(self, target):
        """ mv target from GPU device into host. """
        if type(target) is cp.ndarray:
            return target.get()

        if type(target) is weakref.ref:
            return target().get()

        raise("The given value not in GPU.")

    def _var_normalized(self, target):
        if type(target) is weakref.ref:
            return target()
        
        if type(target) is np.ndarray:
            wr_target = self.htod(target)
            return wr_target()

        return target

    def _result_normalized(self, result, gpu_out):
        if gpu_out:
            return result
        else:
            self.mempool_used.append(result)
            wr_result = weakref.ref(result)
            return wr_result

    def dot(self, A, B, gpu_out: bool = True):
        gpu_A = self._var_normalized(A)
        gpu_B = self._var_normalized(B)

        result = GPUCalculator.dot(gpu_A, gpu_B, gpu_out)

        return self._result_normalized(result, gpu_out)

    def tensor(self, A, B, gpu_out: bool = True):
        gpu_A = self._var_normalized(A)
        gpu_B = self._var_normalized(B)

        result = GPUCalculator.tensor(gpu_A, gpu_B, gpu_out)

        return self._result_normalized(result, gpu_out)

    def MatrixTensorI(self, A, n, m, gpu_out: bool = True):
        gpu_A = self._var_normalized(A)

        result = GPUCalculator.MatrixTensorI(gpu_A, n, m, gpu_out)

        return self._result_normalized(result, gpu_out)

    def VectorPermutation(self, A, mapping, changeInput: bool = False, gpu_out: bool = True):
        gpu_A = self._var_normalized(A)

        result = GPUCalculator.VectorPermutation(gpu_A, mapping, changeInput, gpu_out)

        return self._result_normalized(result, gpu_out)

    def MatrixPermutation(self, A, mapping, changeInput: bool = False, gpu_out: bool = True):
        gpu_A = self._var_normalized(A)

        result = GPUCalculator.MatrixPermutation(gpu_A, mapping, changeInput, gpu_out)

        return self._result_normalized(result, gpu_out)

    def vectordot(self, A, V, mapping, gpu_out: bool = True):
        gpu_A = self._var_normalized(A)
        gpu_V = self._var_normalized(V)

        result = GPUCalculator.vectordot(gpu_A, gpu_V, mapping, gpu_out)

        return self._result_normalized(result, gpu_out)
