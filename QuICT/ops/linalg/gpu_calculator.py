#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import math
import numpy as np
import cupy as cp

from typing import *

from numba import cuda as nb_cuda
import numba as nb


def htod(target):
    """ mv target from host into GPU device. """
    if type(target) is not cp.ndarray:
        return cp.array(target)

    raise (f"The given value has been added in the GPU.")


def dtoh(target):
    """ mv target from GPU device into host. """
    if type(target) is cp.ndarray:
        return target.get()

    raise ("The given value not in GPU.")


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
    assert (A.shape[1] == B.shape[0])

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


matrixt_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_tensorI(const complex<double>* x, complex<double>* y, int n, int m, int rx, int cx) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xid = tid/(n*m);
        int mnid = tid%(n*m);
        int nid = mnid/m;
        int mid = mnid%m;
        int yid = (nid*rx*m + (xid/cx)*m + mid)*cx*m*n + nid*m*cx + (xid%cx)*m + mid;
        y[yid] = x[xid];
    }
    ''', 'matrix_tensorI')


def MatrixTensorI(A, n, m, gpu_out: bool =True):
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

    if n == 1 and m == 1:
        gpu_result = gpu_A
    else:
        gpu_result = cp.zeros((row_a*n*m, col_a*n*m), dtype=np.complex64)
        core_number = gpu_A.size*n*m
        matrixt_kernel((math.ceil(core_number/1024),), (min(1024, core_number),), (gpu_A, gpu_result, cp.int32(n), cp.int32(m), cp.int32(row_a), cp.int32(col_a)))

    if gpu_out:
        return gpu_result.get()

    return gpu_result


def _reindex_by_mapping(mapping):
    n = mapping.shape[0]
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

    out = cp.empty_like(gpu_A)
    for i in range(gpu_A.ndim):
        cp.take(gpu_A, gpu_reidx, axis=i, out=out)

    if changeInput:
        A[:] = out.get() if type(A) is np.ndarray else out

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
        A[:, :] = out_lvl2.get() if type(A) is np.ndarray else out_lvl2

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
    assert (row_a == 1 << mapping.shape[0])

    # Data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_V = cp.array(V) if type(V) is np.ndarray else V

    # matrix permutation for A depending on mapping
    argsorted_mapping = np.argsort(mapping)
    m_array = np.arange(m, dtype=np.int32)

    if not np.allclose(argsorted_mapping, m_array):
        gpu_A = MatrixPermutation(A, argsorted_mapping, changeInput=False, gpu_out=False)

    # generate reindex
    gpu_idx = cp.arange(1 << n, dtype=np.int64)
    gpu_reidx = cp.zeros(int(1 << n), dtype=np.int64)

    mapping_num, remaining_num = 0, 0
    for i in range(n):
        if i in mapping:
            gpu_reidx = cp.add(gpu_reidx,
                               cp.left_shift(cp.bitwise_and(cp.right_shift(gpu_idx, i), 1), n - m + mapping_num))
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


# state vector is in reversed qubit order
@nb_cuda.jit()
def _small_mat_large_vec_kernel(
        small_mat: cp.ndarray,
        large_vec: cp.ndarray,
        qubit_num: int,
        affect_args: cp.ndarray,
        affect_args_sorted: cp.ndarray,
        offset: int,
        upper_bound: int,
):
    bx = nb_cuda.blockIdx.x
    tx = nb_cuda.threadIdx.x
    bw = nb_cuda.blockDim.x
    label = offset + bx * bw + tx
    if label >= upper_bound:
        return

    mat_sz = small_mat.shape[0]
    affect_num = affect_args.shape[0]

    indices = nb_cuda.local.array(shape=64, dtype=nb.types.int64)

    # insert affect_args into corresponding positions
    indices[0] = label
    cur_bit_width = qubit_num - affect_num
    for i in range(affect_num):
        tail_len = cur_bit_width - affect_args_sorted[i]
        mask = (1 << tail_len) - 1
        tail = indices[0] & mask
        indices[0] >>= tail_len
        indices[0] <<= tail_len + 1
        indices[0] |= tail
        cur_bit_width += 1
    for i in range(affect_num):
        n = 1 << i
        mask = 1 << (qubit_num - affect_args[affect_num - 1 - i] - 1)
        for j in range(n):
            indices[n + j] = (indices[j] | mask)

    tmp_vec_cache = nb_cuda.local.array(shape=64, dtype=nb.types.complex64)

    for i in range(mat_sz):
        idx = indices[i]
        tmp_vec_cache[i] = large_vec[idx]
        large_vec[idx] = 0.0 + 0.0j
    for i in range(mat_sz):
        for j in range(mat_sz):
            large_vec[indices[i]] += small_mat[i, j] * tmp_vec_cache[j]


# def _small_mat_large_vec_kernel_sim(
#         block_id: int,
#         thread_id: int,
#         block_size: int,
#         small_mat: cp.ndarray,
#         large_vec: cp.ndarray,
#         qubit_num: int,
#         affect_args: cp.ndarray,
#         affect_args_sorted: cp.ndarray,
#         offset: int,
#         upper_bound: int,
# ):
#     bx = block_id
#     tx = thread_id
#     bw = block_size
#     label = offset + bx * bw + tx
#     if label >= upper_bound:
#         return
#
#     mat_sz = small_mat.shape[0]
#     affect_num = affect_args.shape[0]
#
#     indices = cp.empty(shape=64, dtype=np.int64)
#
#     # insert affect_args into corresponding positions
#     indices[0] = label
#     cur_bit_width = qubit_num - affect_num
#     for i in range(affect_num):
#         tail_len = cur_bit_width - affect_args_sorted[i]
#         mask = (1 << tail_len) - 1
#         tail = indices[0] & mask
#         indices[0] >>= tail_len
#         indices[0] <<= tail_len + 1
#         indices[0] |= tail
#         cur_bit_width += 1
#     for i in range(affect_num):
#         n = 1 << i
#         mask = 1 << (qubit_num - affect_args[affect_num - 1 - i] - 1)
#         for j in range(n):
#             indices[n + j] = (indices[j] | mask)
#
#     tmp_vec_cache = cp.empty(shape=64, dtype=np.complex64)
#
#     for i in range(mat_sz):
#         idx = indices[i]
#         tmp_vec_cache[i] = large_vec[idx]
#         large_vec[idx] = 0.0 + 0.0j
#     for i in range(mat_sz):
#         for j in range(mat_sz):
#             large_vec[indices[i]] += small_mat[i, j] * tmp_vec_cache[j]
#     return 1


def vector_dot_cuda(
        small_mat_,
        large_vec_,
        affect_args_,
        affect_args_sorted_,
):
    thread_per_block = 256
    block_num = (large_vec_.shape[0] + thread_per_block - 1) // thread_per_block
    qubit_num = np.log2(large_vec_.shape[0]).astype(np.int64)
    remained_vec_index = large_vec_.shape[0] // small_mat_.shape[0]

    _small_mat_large_vec_kernel[block_num, thread_per_block](
        small_mat_,
        large_vec_,
        qubit_num,
        affect_args_,
        affect_args_sorted_,
        0,
        remained_vec_index,
    )


# def vector_dot_cuda_sim(
#         small_mat_: cp.ndarray,
#         large_vec_: cp.ndarray,
#         affect_args_: cp.ndarray,
# ):
#     thread_per_block = 256
#     qubit_num = np.log2(large_vec_.shape[0]).astype(np.int64)
#     remained_vec_index = large_vec_.shape[0] // small_mat_.shape[0]
#     block_num = (remained_vec_index + thread_per_block - 1) // thread_per_block
#     affect_args_sorted = affect_args_.copy()
#     affect_args_sorted = cp.sort(affect_args_sorted)
#     result = cp.array(large_vec_)
#     for b in range(block_num):
#         for t in range(thread_per_block):
#             _small_mat_large_vec_kernel_sim(
#                 b,
#                 t,
#                 thread_per_block,
#                 cp.array(small_mat_),
#                 result,
#                 qubit_num,
#                 cp.array(affect_args_),
#                 cp.array(affect_args_sorted),
#                 0,
#                 remained_vec_index,
#             )
#     large_vec_[:] = result.get()[:]
#     return 1


def VectorPermutationRaw(A, mapping, changeInput: bool = False, gpu_out: bool = True):
    """ permutaion A with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A.
        mapping(np.array<int>): the qubit mapping.
        changeInput(bool): whether changes in A.
        gpu_out(bool): return result from GPU.

    Returns:
        np.array<np.complex>: the result of Permutation
    """
    gpu_A = A
    gpu_reidx = _reindex_by_mapping(mapping)

    # out = cp.empty_like(gpu_A) if not changeInput else gpu_A
    out = cp.empty_like(gpu_A)
    cp.take(gpu_A, gpu_reidx, axis=0, out=out)

    if changeInput:
        gpu_A[:] = out[:]

    if gpu_out:
        return out.get()

    return out


def _get_vec_mapping(qubit_num, affect_args_):
    if type(affect_args_) is cp.ndarray:
        affect_args = dtoh(affect_args_)
    else:
        affect_args = affect_args_
    occ_indicator = np.zeros(qubit_num, dtype=bool)
    for bit in affect_args:
        occ_indicator[bit] = True
    tail = 0
    vec_mapping = np.empty(qubit_num, dtype=np.int32)
    vec_mapping_inv = np.empty(qubit_num, dtype=np.int32)
    for i in range(qubit_num):
        if not occ_indicator[i]:
            vec_mapping[tail] = i
            tail += 1
    vec_mapping[tail:] = affect_args
    for i in range(qubit_num):
        vec_mapping_inv[vec_mapping[i]] = i
    return vec_mapping, vec_mapping_inv


def vector_dot_refined(
        small_mat_: cp.ndarray,
        large_vec_: cp.ndarray,
        affect_args_: cp.ndarray,
        gpu_out_: bool = True,
):
    # small_mat: cp.ndarray = cp.array(small_mat_) if type(small_mat_) is np.ndarray else small_mat_
    # large_vec: cp.ndarray = cp.array(large_vec_) if type(large_vec_) is np.ndarray else large_vec_
    small_mat = small_mat_
    large_vec = large_vec_

    qubit_num = np.log2(large_vec_.shape[0]).astype(np.int32)
    affect_num = affect_args_.shape[0]

    vec_mapping, vec_mapping_inv = _get_vec_mapping(qubit_num, affect_args_)

    VectorPermutationRaw(large_vec, vec_mapping_inv, True, False)
    large_vec_mat_form = large_vec.reshape(1 << affect_num, 1 << (qubit_num - affect_num), order='F')
    large_vec_mat_form = cp.dot(small_mat, large_vec_mat_form)
    large_vec = large_vec_mat_form.reshape(1 << qubit_num, order='F')
    VectorPermutationRaw(large_vec, vec_mapping, True, False)
    if gpu_out_:
        return large_vec.get()
    else:
        return large_vec
