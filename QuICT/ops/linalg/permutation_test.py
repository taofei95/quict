import time

from numba import njit, prange
import numpy as np

from unitary_calculation import *

def log_time(f, func, A, B, name = None, w = False):
    if name is None:
        name = func.__name__
    t1 = time.time()
    C = func(A, B)
    t2 = time.time()
    f.write(f"{func.__name__} {name} time pre jit:{(t2 - t1)}\n")
    t1 = time.time()
    C = func(A, B)
    t2 = time.time()
    f.write(f"{func.__name__} {name} time after jit:{t2 - t1}\n")
    if w:
        f.write(f"{list(C)}\n")

def BasedMatrixPermutation(A, mapping, inplace = False):
    """ permutaion A with mapping, inplace

    2^n * 2^n matrix, n qubits

    Args:
        A(np.array<np.complex>): the matrix A
        mapping(list<int>): the qubit mapping
        inplace(bool): whether changes in A
    Returns:
        np.array<np.complex>: the result of Permutation
    """
    n = A.shape[0]
    assert(2**len(mapping) == n)

    switched_idx = np.arange(n, dtype=np.int64)

    for idx, new_idx in enumerate(mapping):
        if idx != new_idx:
            if new_idx < idx and mapping[new_idx] == idx:
                continue
            Based_bitswitch(switched_idx, idx, new_idx)

    if not inplace:
        out = A[(switched_idx),:].copy()
        return out[:,(switched_idx)]

    default_idx = np.arange(n, dtype=np.int64)
    A[(default_idx),:] = A[(switched_idx),:]
    A[:,(default_idx)] = A[:,(switched_idx)]

def BasedVectorPermutation(A, mapping, inplace = False):
    """ permutaion A with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A
        mapping(list<int>): the qubit mapping
        inplace(bool): whether changes in A
    Returns:
        np.array<np.complex>: the result of Permutation
    """
    n = A.shape[0]
    assert(2**len(mapping) == n)

    switched_idx = np.arange(n, dtype=np.int64)

    for idx, new_idx in enumerate(mapping):
        if idx != new_idx:
            if new_idx < idx and mapping[new_idx] == idx:
                continue
            Based_bitswitch(switched_idx, idx, new_idx)

    if not inplace:
        return A[(switched_idx),].copy()

    A[(np.arange(n, dtype=np.int64)),] = A[(switched_idx),]

def Based_bitswitch(index, i: int, j: int):
    """ switch bits in position i and j

    Args:
        A(np.array<np.int>): the array of index
        i(int): the switched bit position
        j(int): the switched bit position
    """
    coef = np.bitwise_or(np.left_shift(1, i), np.left_shift(1, j))
    w = (np.bitwise_and(np.right_shift(index, i), 1) != np.bitwise_and(np.right_shift(index, j), 1))

    np.bitwise_xor(index, coef, where=w, out=index)

for i in range(7, 15):
    w = False
    f_tI = open(f"vector_permutation_qubit_r_{i}.txt", "w")
    n = 1 << i
    A = np.random.random((n,n))
    if i <= 10:
        f_tI.write(f"{list(A)}\n")
        f_tI.write("------------------\n")
        w = True
    mapping = list(range(i))[::-1]
    mapping = np.array(mapping)
    np.random.shuffle(mapping)
    f_tI.write(f"{list(mapping)}\n")
    f_tI.write("------------------\n")
    log_time(f_tI, VectorPermutation, A, mapping, "permutation_with_numba")
    log_time(f_tI, BasedVectorPermutation, A, mapping, "permutation_without_numba", w)
