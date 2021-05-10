import math
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
from pycuda.compiler import SourceModule
import time

from utils import mapping_augment


DOT_TEMPLATE = SourceModule(r"""
    #include <pycuda-complex.hpp>
    __global__ void dot(pycuda::complex<double> *out, const pycuda::complex<double> *A, const pycuda::complex<double> *B, const int *size)
    {
        const int out_x = size[0];
        const int out_y = size[3];

        const int stride_x = blockDim.x;
        const int stride_y = blockDim.y;

        const int idx_x = threadIdx.x;
        const int idx_y = threadIdx.y;

        for (int i = idx_x; i < out_x; i += stride_x){
            for (int j = idx_y; j < out_y; j += stride_y){
                pycuda::complex<double> As = 0;
                for (int z = 0; z < size[1]; z++){
                    As += A[i*size[1] + z] * B[z*out_y + j];
                }
                out[i*out_y+j] = As;
            }
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

        const int stride_x = blockDim.x * gridDim.y;
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


class GPUCalculator:
    """ Based matrix algorithms for running in GPU. """
    @staticmethod
    def dot(A, B, gpu_in: bool = True, gpu_out: bool = True):
        """ dot matrix A and matrix B

        Args:
            A(np.array<np.complex>): the matrix A
            B(np.array<np.complex>): the matrix B

        Returns:
            np.array<np.complex>: A * B
        """
        row_a, col_a = A.shape
        row_b, col_b = B.shape
        assert(col_a == row_b)

        if gpu_in:
            gpu_A = gpuarray.to_gpu(A)
            gpu_B = gpuarray.to_gpu(B)
        else:
            gpu_A = A
            gpu_B = B

        gpu_out = gpuarray.zeros((row_a, col_b), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, row_b, col_b], dtype=np.int32))

        gpu_dot = DOT_TEMPLATE.get_function("dot")

        block = (min(row_a, 32), min(col_b, 32), 1)
        gpu_dot(gpu_out, gpu_A, gpu_B, gpu_size, block=block)

        if gpu_out:
            return gpu_out.get()

        return gpu_out

    @staticmethod
    def tensor(A, B, gpu_in: bool = True, gpu_out: bool = True):
        """ tensor A and B

        Args:
            A(np.array<np.complex>): the matrix A
            B(np.array<np.complex>): the matrix B

        Returns:
            np.array<np.complex>: the tensor result A ⊗ B
        """
        row_a, col_a = A.shape
        row_b, col_b = B.shape

        if gpu_in:
            gpu_A = gpuarray.to_gpu(A)
            gpu_B = gpuarray.to_gpu(B)
        else:
            gpu_A = A
            gpu_B = B

        gpu_out = gpuarray.zeros((row_a*row_b, col_a*col_b), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, row_b, col_b], dtype=np.int32))

        gpu_tensor = TENSOR_TEMPLATE.get_function("tensor")

        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (min(row_b, 32), min(col_b, 32))
        gpu_tensor(gpu_out, gpu_A, gpu_B, gpu_size, grid=grid, block=block)

        if gpu_out:
            return gpu_out.get()

        return gpu_out

    @staticmethod
    def matrix_tensor(A, n, m, gpu_in: bool = True, gpu_out: bool = True):
        """ tensor I^n and A and I^m

        Args:
            A(np.array<np.complex>): the matrix A
            n(int): the index of indentity
            m(int): the index of indentity
            gpu_in(bool): put data into GPU
            gpu_out(bool): return CPU data

        Returns:
            np.array<np.complex>: the tensor result I^n ⊗ A ⊗ I^m
        """
        if gpu_in:
            gpu_A = gpuarray.to_gpu(A)
        else:
            gpu_A = A

        row_a, col_a = A.shape
        gpu_out = gpuarray.zeros((n*m*A.shape[0], n*m*A.shape[1]), dtype=np.complex_)
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, n, m], dtype=np.int32))

        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (min(n, 32), min(m, 32))
        gpu_tensorM = TENSOR_MATRIX_TEMPLATE.get_function("tensor_matrix")
        gpu_tensorM(gpu_out, gpu_A, gpu_size, grid=grid, block=block)
        
        if gpu_out:
            return gpu_out.get()

        return gpu_out

    @staticmethod
    def vector_permutation(A, mapping, inplace, gpu_in: bool = True, gpu_out: bool = True):
        """ permutaion A with mapping, inplace

        Args:
            A(np.array<np.complex>): the matrix A
            mapping(list<int>): the qubit mapping
            inplace(bool): whether changes in A
        Returns:
            np.array<np.complex>: the result of Permutation
        """
        mapping = np.array(mapping, dtype=np.int64)

        if not A.shape[0] == 1 << mapping.shape[0]:
            raise IndexError("Indices do not match!")

        if gpu_in:
            gpu_A = gpuarray.to_gpu(A)
        else:
            gpu_A = A

        idx_mapping = mapping_augment(mapping)
        gpu_idx = gpuarray.to_gpu(idx_mapping)

        if not inplace:
            return gpuarray.take(gpu_A, gpu_idx).get() if gpu_out else gpuarray.take(gpu_A, gpu_idx)
        
        A = gpuarray.take(gpu_A, gpu_idx).get() if gpu_out else gpuarray.take(gpu_A, gpu_idx)

    @staticmethod
    def matrix_permutation(A, mapping, inplace, gpu_in: bool = True, gpu_out: bool = True):
        """ permute mat with mapping, inplace

        Args:
            mat: Matrix to be permuted.
            mapping: An array-like object indicating bit ordering.
            inplace: Whether change the input matrix.
        """
        mapping = np.array(mapping, dtype=np.int64)

        if not A.shape[0] == 1 << mapping.shape[0]:
            raise IndexError("Indices do not match!")

        if gpu_in:
            gpu_A = gpuarray.to_gpu(A)
        else:
            gpu_A = A

        idx_mapping = mapping_augment(mapping)
        gpu_idx = gpuarray.to_gpu(idx_mapping)

        if not inplace:
            return gpuarray.take(gpu_A, gpu_idx).get() if gpu_out else gpuarray.take(gpu_A, gpu_idx)
        
        A = gpuarray.take(gpu_A, gpu_idx).get() if gpu_out else gpuarray.take(gpu_A, gpu_idx)
