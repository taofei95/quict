#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .utils import mapping_augment


DOT_TEMPLATE = SourceModule(r"""
    #include <pycuda-complex.hpp>
    __global__ void dot(pycuda::complex<double> *out, const pycuda::complex<double> *A, const pycuda::complex<double> *B, const int *size)
    {
        const int out_x = size[0];
        const int out_y = size[3];
        const int mul = size[1];

        const int stride = blockDim.x;

        const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

        pycuda::complex<double> C = 0;
        for (int i = 0; i < gridDim.x; i++){
            for (int k = 0; k < stride; k++){
                C += A[idx_x*mul + i*stride + k] * B[(i*blockDim.x + k)*out_y + idx_y];
            }
        }
        if ((idx_x < out_x) && (idx_y < out_y)){
            out[idx_x*out_y + idx_y] = C;
        }
    }
    """)

TENSOR_TEMPLATE = SourceModule(r"""
    #include <pycuda-complex.hpp>
    __global__ void tensor(pycuda::complex<double> *out, const pycuda::complex<double> *A, const pycuda::complex<double> *B, const int *size)
    {
        const int out_x = size[0] * size[2];
        const int out_y = size[1] * size[3];

        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;

        const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

        for (int i = idx_x; i < out_x; i += stride_x)
        {
            for (int j = idx_y; j < out_y; j += stride_y)
            {
                out[i*out_y+j] = A[(i/size[2])*size[1]+j/size[3]] * B[(i % size[2])*size[3]+(j % size[3])];
            }
        }
    }
    """)

TENSOR_MATRIX_TEMPLATE = SourceModule(r"""
    #include <pycuda-complex.hpp>
    void __global__ tensor_matrix(pycuda::complex<double> *out, const pycuda::complex<double> *A, const int *size)
    {
        const int n = size[2];
        const int m = size[3];

        const int out_x = n * m * size[0];
        const int out_y = n * m * size[1];

        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;

        const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

        for(int i = idx_x; i < out_x; i += stride_x){
            for(int j = idx_y; j < out_y; j += stride_y){
                if (i / (m * size[0]) == j / (m * size[1])){
                    int r_x = i % (m * size[0]);
                    int r_y = j % (m * size[1]);
                    if ((r_x == r_y) || (r_x % m) == r_y || r_x == (r_y % m) || (r_x % m) == (r_y % m)){
                        out[i*out_y+j] = A[r_x / m * size[1] + r_y / m];
                    }
                }
            }
        }
    }
    """)

MATRIX_PERM_TEMPLATE = SourceModule(r"""
    #include <pycuda-complex.hpp>
    void __global__ perm_matrix(pycuda::complex<double> *out, const pycuda::complex<double> *A, const int *mapping, const int *size)
    {
        const int out_x = size[0];
        const int out_y = size[1];

        const int stride_x = blockDim.x * gridDim.x;
        const int stride_y = blockDim.y * gridDim.y;

        const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
        const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

        for(int i = idx_x; i < out_x; i += stride_x){
            for(int j = idx_y; j < out_y; j += stride_y){
                const int reverse_x = mapping[i];
                const int reverse_y = mapping[j];
                out[i * out_y + j] = A[reverse_x * out_y + reverse_y];
            }
        }
    }
""")


class GPUCalculator:
    """ Based matrix algorithms for running in GPU. """
    def __init__(self, gpu_device: int = 0):
        self.gpu_device = gpu_device

        if self.gpu_device != 0:
            os.environ["CUDA_DEVICE"] = str(self.gpu_device)

        cuda.init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_device != 0:
            os.environ["CUDA_DEVICE"] = "0"

    @staticmethod
    def htod(target):
        if type(target) is not gpuarray.GPUArray:
            return gpuarray.to_gpu(target)

        raise(f"The given value has been added in the GPU.")

    @staticmethod
    def dot(A, B, gpu_out: bool = True):
        """ dot matrix A and matrix B

        Args:
            A(np.array<np.complex>): the matrix A
            B(np.array<np.complex>): the matrix B
            gpu_out(bool): return result from GPU into CPU

        Returns:
            np.array<np.complex>: A * B
        """
        row_a, col_a = A.shape
        row_b, col_b = B.shape
        assert(col_a == row_b)

        # Data in GPU.
        gpu_A = gpuarray.to_gpu(A) if type(A) is np.ndarray else A
        gpu_B = gpuarray.to_gpu(B) if type(B) is np.ndarray else B

        gpu_result = gpuarray.zeros((row_a, col_b), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, row_b, col_b], dtype=np.int32))

        # GPU kernel function.
        gpu_dot = DOT_TEMPLATE.get_function("dot")

        block = (min(row_b, 32), min(row_b, 32), 1)
        grid = (max(row_a//block[0], 1), max(col_b//block[0], 1), 1)
        gpu_dot(gpu_result, gpu_A, gpu_B, gpu_size, grid=grid, block=block)

        if gpu_out:
            return gpu_result.get()

        return gpu_result

    @staticmethod
    def tensor(A, B, gpu_out: bool = True):
        """ tensor A and B

        Args:
            A(np.array<np.complex>): the matrix A
            B(np.array<np.complex>): the matrix B
            gpu_out(bool): return result from GPU into CPU

        Returns:
            np.array<np.complex>: the tensor result A ⊗ B
        """
        row_a, col_a = A.shape
        row_b, col_b = B.shape

        # Data in GPU.
        gpu_A = gpuarray.to_gpu(A) if type(A) is np.ndarray else A
        gpu_B = gpuarray.to_gpu(B) if type(B) is np.ndarray else B

        gpu_result = gpuarray.zeros((row_a*row_b, col_a*col_b), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, row_b, col_b], dtype=np.int32))

        # GPU kernel function.
        gpu_tensor = TENSOR_TEMPLATE.get_function("tensor")

        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (min(row_b, 32), min(col_b, 32))
        gpu_tensor(gpu_result, gpu_A, gpu_B, gpu_size, grid=grid, block=block)

        if gpu_out:
            return gpu_result.get()

        return gpu_result

    @staticmethod
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
        gpu_A = gpuarray.to_gpu(A) if type(A) is np.ndarray else A
        gpu_result = gpuarray.zeros((n*m*A.shape[0], n*m*A.shape[1]), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, n, m], dtype=np.int32))

        # GPU kernel function
        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (min(n, 32), min(m, 32))
        gpu_tensorM = TENSOR_MATRIX_TEMPLATE.get_function("tensor_matrix")
        gpu_tensorM(gpu_result, gpu_A, gpu_size, grid=grid, block=block)

        if gpu_out:
            return gpu_result.get()

        return gpu_result

    @staticmethod
    def VectorPermutation(A, mapping, changeInput: bool = False, gpu_out: bool = True):
        """ permutaion A with mapping, inplace

        Args:
            A(np.array<np.complex>): the matrix A.
            mapping(list<int>): the qubit mapping.
            changeInput(bool): whether changes in A.
            gpu_out(bool): return result from GPU.
        Returns:
            np.array<np.complex>: the result of Permutation
        """
        mapping = np.array(mapping, dtype=np.int64)

        if not A.shape[0] == 1 << mapping.shape[0]:
            raise IndexError("Indices do not match!")

        idx_mapping = mapping_augment(mapping)

        # data in GPU
        gpu_A = gpuarray.to_gpu(A) if type(A) is np.ndarray else A
        gpu_idx = gpuarray.to_gpu(idx_mapping)

        if not changeInput:
            return gpuarray.take(gpu_A, gpu_idx).get() if gpu_out else gpuarray.take(gpu_A, gpu_idx)

        A = gpuarray.take(gpu_A, gpu_idx).get() if gpu_out else gpuarray.take(gpu_A, gpu_idx)

    @staticmethod
    def MatrixPermutation(A, mapping, changeInput: bool = False, gpu_out: bool = True):
        """ permute mat with mapping, inplace

        Args:
            A: Matrix to be permuted.
            mapping: An array-like object indicating bit ordering.
            changeInput: Whether change the input matrix.
            gpu_out(bool): return result from GPU into CPU.
        """
        row_a, col_a = A.shape
        if not row_a == 1 << mapping.shape[0]:
            raise IndexError("Indices do not match!")

        # generate new idx depending on given mapping
        idx_mapping = mapping_augment(mapping)

        # data in GPU
        gpu_A = gpuarray.to_gpu(A) if type(A) is np.ndarray else A
        gpu_idx = gpuarray.to_gpu(idx_mapping.astype(np.int32))
        gpu_result = gpuarray.empty((row_a, col_a), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a], dtype=np.int32))

        # GPU kernel function
        gpu_permM = MATRIX_PERM_TEMPLATE.get_function("perm_matrix")

        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (max(row_a//block[0], 1), max(col_a//block[1], 1))
        gpu_permM(gpu_result, gpu_A, gpu_idx, gpu_size, grid=grid, block=block)

        if not changeInput:
            return gpu_result.get().reshape(row_a, col_a) if gpu_out else gpu_result

        A = gpu_result.get().reshape(row_a, col_a) if gpu_out else gpu_result

