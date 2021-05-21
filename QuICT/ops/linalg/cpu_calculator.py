#!/usr/bin/env python
# -*- coding:utf8 -*-

from numba import jit, njit, prange
import numpy as np
from typing import *

from .utils import mapping_augment



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

@njit(parallel=True, nogil=True)
def MatrixPermutation(A: np.ndarray, mapping: np.ndarray, changeInput: bool = False) -> np.ndarray:
    """ permute A with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A
        mapping(np.ndarray<int>): the qubit mapping
        changeInput(bool): whether changes in A

    """
    if not A.shape[0] == (1 << mapping.shape[0]):
        raise IndexError("Indices do not match!")

    idx_mapping = mapping_augment(mapping)

    # Do NOT perform parallel operations over row permutations!
    # They are just too spare in memory. Elements in the same column
    # are distributed with a gap as matrix row length.
    perm_mat = A[idx_mapping, :]
    for i in prange(idx_mapping.shape[0]):
        perm_mat[i] = perm_mat[i][idx_mapping]

    if changeInput:
        A[:, :] = perm_mat

    return perm_mat

@njit()
def VectorPermutation(A: np.ndarray, mapping: np.ndarray, changeInput: bool = False):
    """ permutaion A with mapping, changeInput

    Args:
        A(np.array<np.complex>): the matrix A
        mapping(np.ndarray<int>): the qubit mapping
        changeInput(bool): whether changes in A
    Returns:
        np.array<np.complex>: the result of Permutation
    """
    if not A.shape[0] == 1 << mapping.shape[0]:
        raise IndexError("Indices do not match!")

    switched_idx = mapping_augment(mapping)

    if changeInput:
        A[:] = A[switched_idx]

    return A[switched_idx]

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
def dot(A, B):
    """ dot matrix A and matrix B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B

    Returns:
        np.array<np.complex>: A * B
    """
    return np.dot(A, B)

def vectordot(A, V, mapping):
    row_a, col_a = A.shape
    n, m = np.log2(V.shape[0]).astype(np.int32), np.log2(row_a).astype(np.int32)
    assert (m == mapping.shape[0])

    # matrix permutation for A depending on mapping
    argsorted_mapping = np.argsort(mapping)

    A_reidx = A
    if not (argsorted_mapping == np.arange(mapping.shape[0])).all():
        A_reidx = MatrixPermutation(A, argsorted_mapping, changeInput=False)

    return _helper_vdot(A_reidx, V, n, m)

@njit()
def _helper_vdot(A, V, n, m):
    # grep m qubit idx depending on mapping
    A_tensor = np.kron(np.identity(1 << n - m).astype(np.complex_), A)

    return np.dot(A_tensor, V)
