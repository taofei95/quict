#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/15 8:47 下午
# @Author  : Han Yu
# @File    : unitary_calculation

from numba import jit, njit, prange
import numpy as np
from typing import *

from cpu_calculator import CPUCalculator
from utils import gpu_decorator, GPU_AVAILABLE


if GPU_AVAILABLE:
    from gpu_calculator import GPUCalculator
    gpu_calculator = GPUCalculator()
else:
    gpu_calculator = CPUCalculator()


MTENSOR_THRESHOLD = 8
TENSOR_THRESHOLD = 9
DOT_THRESHOLD = 10
MPERM_THRESHOLD = 8
VPERM_THRESHOLD = 14


@gpu_decorator(threshold=MTENSOR_THRESHOLD, cpu_func=CPUCalculator.MatrixTensorI, gpu_func=gpu_calculator.MatrixTensorI)
def MatrixTensorI(A, n, m, gpu_in: bool = True, gpu_out: bool = False) -> np.ndarray:
    """ tensor I^n and A and I^m

    Args:
        A(np.array<np.complex>): the matrix A
        n(int): the index of indentity
        m(int): the index of indentity
        (below variables only available in gpu_function)
        * gpu_in(bool): mv data from CPU into GPU
        * gpu_out(bool): return result from GPU into CPU

    Returns:
        np.array<np.complex>: the tensor result I^n ⊗ A ⊗ I^m
    """
    pass


@gpu_decorator(threshold=MPERM_THRESHOLD, cpu_func=CPUCalculator.MatrixPermutation, gpu_func=gpu_calculator.MatrixPermutation)
def MatrixPermutation(
    A: np.ndarray,
    mapping: np.ndarray,
    changeInput: bool = False,
    gpu_in: bool = True,
    gpu_out: bool = False
) -> np.ndarray:
    """ permute A with mapping, inplace

    Args:
        A: Matrix to be permuted.
        mapping(np.ndarray): An array-like object indicating bit ordering.
        changeInput: Whether change the input matrix.
        (below variables only available in gpu_function)
        * gpu_in(bool): mv data from CPU into GPU
        * gpu_out(bool): return result from GPU into CPU

    """
    pass


@gpu_decorator(threshold=VPERM_THRESHOLD, cpu_func=CPUCalculator.VectorPermutation, gpu_func=gpu_calculator.VectorPermutation)
def VectorPermutation(
    A: np.ndarray,
    mapping: np.ndarray,
    changeInput: bool = False,
    gpu_in: bool = True,
    gpu_out: bool = False
) -> np.ndarray:
    """ permutaion A with mapping, inplace

    Args:
        A(np.array<np.complex>): the vector A
        mapping(np.ndarray): the qubit mapping
        changeInput(bool): whether changes in A
        (below variables only available in gpu_function)
        * gpu_in(bool): mv data from CPU into GPU
        * gpu_out(bool): return result from GPU into CPU
        
    Returns:
        np.array<np.complex>: the result of Permutation
    """
    pass


@gpu_decorator(threshold=TENSOR_THRESHOLD, cpu_func=CPUCalculator.tensor, gpu_func=gpu_calculator.tensor)
def tensor(A, B, gpu_in: bool = True, gpu_out: bool = False):
    """ tensor A and B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B
        (below variables only available in gpu_function)
        * gpu_in(bool): mv data from CPU into GPU
        * gpu_out(bool): return result from GPU into CPU

    Returns:
        np.array<np.complex>: the tensor result A ⊗ B
    """
    pass

@gpu_decorator(threshold=DOT_THRESHOLD, cpu_func=CPUCalculator.dot, gpu_func=gpu_calculator.dot)
def dot(A, B, gpu_in: bool = True, gpu_out: bool = False):
    """ dot matrix A and matrix B

    Args:
        A(np.array<np.complex>): the matrix A
        B(np.array<np.complex>): the matrix B
        (below variables only available in gpu_function)
        * gpu_in(bool): mv data from CPU into GPU
        * gpu_out(bool): return result from GPU into CPU

    Returns:
        np.array<np.complex>: A * B
    """
    pass
