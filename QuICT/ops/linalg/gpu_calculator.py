#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import math
import numpy as np
import cupy as cp


def htod(target):
    """ mv target from host into GPU device. """
    if type(target) is not cp.ndarray:
        return cp.array(target)

    raise(f"The given value has been added in the GPU.")


def dtoh(target):
    """ mv target from GPU device into host. """
    if type(target) is cp.ndarray:
        return target.get()

    raise("The given value not in GPU.")


def flush_memory():
    """ Release unused memory in current GPU device. """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def dot(A, B, gpu_out: bool = True):
    """ dot matrix A and matrix B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B
        gpu_out(bool): return result from GPU into CPU

    Returns:
        np.array<np.complex>: A * B
    """
    assert(A.shape[1]  == B.shape[0])

    # Data in GPU.
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_B = cp.array(B) if type(B) is np.ndarray else B

    return cp.dot(gpu_A, gpu_B).get() if gpu_out else cp.dot(gpu_A, gpu_B)


def tensor(A, B, gpu_out: bool = True):
    """ tensor A and B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B
        gpu_out(bool): return result from GPU into CPU

    Returns:
        np.array<np.complex>: the tensor result A ⊗ B
    """
    # Data in GPU.
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_B = cp.array(B) if type(B) is np.ndarray else B

    if gpu_out:
        return cp.kron(gpu_A, gpu_B).get()

    return cp.kron(gpu_A, gpu_B)


def MatrixTensorI(A, n, m, gpu_out: bool = True):
    """ tensor I^n and A and I^m

    Args:
        A(np.array<np.complex>): the matrix A
        n(int): the index of indentity
        m(int): the index of indentity
        gpu_out(bool): return result from GPU into CPU

    Returns:
        np.array<np.complex>: the tensor result I^n ⊗ A ⊗ I^m
    """
    row_a, col_a = A.shape

    # Data in GPU.
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_IN = cp.identity(n, dtype=np.complex128)
    gpu_IM = cp.identity(m, dtype=np.complex128)
    gpu_result = cp.kron(cp.kron(gpu_IN, gpu_A), gpu_IM)

    if gpu_out:
        return gpu_result.get()

    return gpu_result


def _reindex_by_mapping(mapping):
    n= mapping.shape[0]
    p2n = 1 << n 

    gpu_idx = cp.arange(p2n, dtype=np.int64)
    gpu_reidx = cp.zeros(p2n, dtype=np.int64)

    for i in range(n):
        right = n - 1 - mapping[i]
        left = n - 1 - i
        cp.bitwise_or(gpu_reidx, cp.left_shift(cp.bitwise_and(cp.right_shift(gpu_idx, right), 1), left), out=gpu_reidx)

    # data clear
    del gpu_idx

    return gpu_reidx


def VectorPermutation(A, mapping, changeInput: bool = False, gpu_out: bool = True):
    """ permutaion A with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A.
        mapping(np.array<int>): the qubit mapping.
        changeInput(bool): whether changes in A.
        gpu_out(bool): return result from GPU.

    Returns:
        np.array<np.complex>: the result of Permutation
    """
    row_a, n = A.shape[0], mapping.shape[0]
    if not row_a == 1 << n:
        raise IndexError("Indices do not match!")

    # data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_reidx = _reindex_by_mapping(mapping)

    out = cp.empty_like(gpu_A) if not changeInput else gpu_A
    for i in range(gpu_A.ndim):
        cp.take(gpu_A, gpu_reidx, axis=i, out=out)

    if gpu_out:
        return out.get()

    return out


def MatrixPermutation(A, mapping, changeInput: bool = False, gpu_out: bool = True):
    """ permute mat with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A.
        mapping(np.array<int>): the qubit mapping.
        changeInput(bool): whether changes in A.
        gpu_out(bool): return result from GPU.
    """
    row_a, col_a = A.shape
    if not row_a == 1 << mapping.shape[0]:
        raise IndexError("Indices do not match!")

    # data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_reidx = _reindex_by_mapping(mapping)

    out_lvl1 = cp.empty_like(gpu_A)
    cp.take(gpu_A, gpu_reidx, axis=0, out=out_lvl1)
    out_lvl2 = cp.empty_like(gpu_A)
    cp.take(out_lvl1, gpu_reidx, axis=1, out=out_lvl2)

    if changeInput:
        gpu_A = out_lvl2

    if gpu_out:
        return out_lvl2.get()

    return out_lvl2


def vectordot(A, V, mapping, gpu_out: bool = True):
    """ dot matrix A and matrix B

    Args:
        A(np.array<np.complex>): the matrix A.
        V(np.array<np.complex>): the vector V.
        mapping(np.array<int>): the qubit mapping.
        gpu_out(bool): return result from GPU into CPU.

    Returns:
        np.array<np.complex>: the vector with length 2^n
    """
    row_a, col_a = A.shape
    n, m = np.log2(V.shape[0]).astype(np.int32), mapping.shape[0]
    assert(row_a == 1 << mapping.shape[0])

    # Data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_V = cp.array(V) if type(V) is np.ndarray else V

    # matrix permutation for A depending on mapping
    argsorted_mapping = np.argsort(mapping)
    m_array = np.arange(m, dtype=np.int32)

    if not np.allclose(argsorted_mapping, m_array):
        gpu_A = MatrixPermutation(A, argsorted_mapping, changeInput = False, gpu_out = False)

    # generate reindex
    gpu_idx = cp.arange(1 << n, dtype=np.int64)
    gpu_reidx = cp.zeros(int(1 << n), dtype=np.int64)

    mapping_num, remaining_num = 0, 0
    for i in range(n):
        if i in mapping:
            gpu_reidx = cp.add(gpu_reidx, cp.left_shift(cp.bitwise_and(cp.right_shift(gpu_idx, i), 1), n - m + mapping_num))
            mapping_num += 1
        else:
            gpu_reidx = cp.add(gpu_reidx, cp.left_shift(cp.bitwise_and(cp.right_shift(gpu_idx, i), 1), remaining_num))
            remaining_num += 1

    gpu_reidx = cp.argsort(gpu_reidx)

    # take V data with new index
    gpu_V_reidx = cp.take(gpu_V, gpu_reidx)

    # GPU dot
    gpu_V_reidx = cp.dot(gpu_A, gpu_V_reidx.reshape(1 << m, -1))

    # put back
    gpu_reidx = cp.argsort(gpu_reidx)
    gpu_result = cp.take(gpu_V_reidx, gpu_reidx)

    if gpu_out:
        return gpu_result.get()

    return gpu_result
