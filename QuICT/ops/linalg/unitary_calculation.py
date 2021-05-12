#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/15 8:47 下午
# @Author  : Han Yu
# @File    : unitary_calculation

from numba import jit, njit, prange
import numpy as np
from typing import *

from .cpu_calculator import CPUCalculator
from .utils import gpu_decorator, GPU_AVAILABLE

@njit(parallel=True, nogil=True)
def MatrixTensorI(A, n, m):
    """ tensor I^n and A and I^m

    Args:
        A(np.array<np.complex>): the matrix A
        n(int): the index of indentity
        m(int): the index of indentity

    Returns:
        np.array<np.complex>: the tensor result I^n ⊗ A ⊗ I^m
    """
    i_m = np.identity(m)
    row_a, col_a = A.shape
    MatrixTensor = np.zeros((n * m * row_a, n * m * col_a), dtype=np.complex_)

    for i in prange(row_a):
        for j in prange(col_a):
            temp_M = A[i, j] * i_m
            for k in range(n):
                start_row_idx = k * m * row_a + i * m
                start_col_idx = k * m * col_a + j * m
                MatrixTensor[start_row_idx:start_row_idx + m, start_col_idx:start_col_idx + m] = temp_M

    return MatrixTensor


@njit(nogil=True)
def mapping_augment(mapping: np.ndarray) -> np.ndarray:
    n = len(mapping)
    p2n = 1 << n
    res = np.zeros(shape=p2n, dtype=np.int64)
    for i in range(p2n):
        for k in range(n):
            res[i] |= ((i >> (n - 1 - mapping[k])) & 1) << (n - 1 - k)
    return res


@njit(parallel=True, nogil=True)
def _innerMatrixPermutation(mat: np.ndarray, mapping: np.ndarray, changeInput=False) -> np.ndarray:
    idx_mapping = mapping_augment(mapping)

    # Do NOT perform parallel operations over row permutations!
    # They are just too spare in memory. Elements in the same column
    # are distributed with a gap as matrix row length.
    perm_mat = mat[idx_mapping, :]
    for i in prange(idx_mapping.shape[0]):
        perm_mat[i] = perm_mat[i][idx_mapping]

    if changeInput:
        mat[:, :] = perm_mat

    return perm_mat


def MatrixPermutation(mat: np.ndarray, mapping: Union[np.ndarray, List[int]], changeInput: bool = False) -> np.ndarray:
    """ permute mat with mapping, inplace

    Args:
        mat: Matrix to be permuted.
        mapping: An array-like object indicating bit ordering.
        changeInput: Whether change the input matrix.

    """

    mapping = np.array(mapping, dtype=np.int64)

    if not mat.shape[0] == 1 << mapping.shape[0]:
        raise IndexError("Indices do not match!")

    return _innerMatrixPermutation(mat, mapping, changeInput)


@njit()
def VectorPermutation(A, mapping, inplace=False):
    """ permutaion A with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A
        mapping(list<int>): the qubit mapping
        inplace(bool): whether changes in A
    Returns:
        np.array<np.complex>: the result of Permutation
    """
    n = A.shape[0]
    assert (2 ** len(mapping) == n)

    switched_idx = IndexRearrangement(mapping)

    if not inplace:
        return A[(switched_idx),].copy()

    A[(np.arange(n, dtype=np.int64)),] = A[(switched_idx),]


@njit()
def IndexRearrangement(mapping):
    """ Depending on the qubit mapping, re-arrange the index

    Args:
        mapping(list<int>): the qubit mapping
    Returns:
        np.array<np.int64>: the array of new index
    """
    n = 2 ** len(mapping)
    switched_idx = np.arange(n, dtype=np.int64)

    for idx, new_idx in enumerate(mapping):
        if idx != new_idx:
            if new_idx < idx and mapping[new_idx] == idx:
                continue
            BitSwitch(switched_idx, idx, new_idx)

    return switched_idx


@njit()
def BitSwitch(index, i: int, j: int):
    """ switch bits in position i and j

    Args:
        A(np.array<np.int>): the array of index
        i(int): the switched bit position
        j(int): the switched bit position
    """
    coef = np.bitwise_or(np.left_shift(1, i), np.left_shift(1, j))
    w = (np.bitwise_and(np.right_shift(index, i), 1) != np.bitwise_and(np.right_shift(index, j), 1))

    index[w] = np.bitwise_xor(index[w], coef)


@njit(parallel=True, nogil=True)
def tensor(A, B):
    """ tensor A and B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B

    Returns:
        np.array<np.complex>: the tensor result A ⊗ B
    """
    row_a, col_a = A.shape
    row_b, col_b = B.shape
    tensor_data = np.empty((row_a * row_b, col_a * col_b), dtype=np.complex_)

    for r in prange(row_a):
        for c in prange(col_a):
            tensor_data[r * row_b:(r + 1) * row_b, c * col_b:(c + 1) * col_b] = A[r, c] * B

    return tensor_data


@njit()
def multiply(A, B):
    """ multiply matrix A and matrix B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B

    Returns:
        np.array<np.complex>: A * B
    """
    return np.dot(A, B)
