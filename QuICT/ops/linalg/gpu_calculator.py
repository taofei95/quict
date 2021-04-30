import math
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
from pycuda.compiler import SourceModule
import time

from utils import mapping_augment


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
                out[j*out_y+i] = A[j/size[3]*size[1]+i/size[2]] * B[j/size[1]*size[3]+i/size[0]];
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
                        out[j*out_y+i] = A[r_y / m * size[1] + r_x / m];
                    }
                }
            }
        }
    }
    """)

class GPUCalculator:
    @staticmethod
    def dot(A, B):
        a = gpuarray.to_gpu(A)
        b = gpuarray.to_gpu(B)
        return gpuarray.dot(a, b).get()

    @staticmethod
    def tensor(A, B):
        row_a, col_a = A.shape
        row_b, col_b = B.shape
        gpu_A = gpuarray.to_gpu(A)
        gpu_B = gpuarray.to_gpu(B)
        gpu_out = gpuarray.to_gpu(np.zeros([row_a*row_b, col_a*col_b], dtype=np.complex_))
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, row_b, col_b], dtype=np.int32))

        gpu_tensor = TENSOR_TEMPLATE.get_function("tensor")

        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (min(row_b, 32), min(col_b, 32))
        gpu_tensor(gpu_out, gpu_A, gpu_B, gpu_size, grid=grid, block=block)

        return gpu_out.get()

    @staticmethod
    def matrix_tensor(A, n, m):
        row_a, col_a = A.shape
        gpu_A = gpuarray.to_gpu(A)
        gpu_out = gpuarray.to_gpu(np.zeros([n*m*A.shape[0], n*m*A.shape[1]], dtype=np.complex_))
        gpu_size = gpuarray.to_gpu(np.array([row_a, col_a, n, m], dtype=np.int32))

        block = (min(row_a, 32), min(col_a, 32), 1)
        grid = (min(n, 32), min(m, 32))
        gpu_tensorM = TENSOR_MATRIX_TEMPLATE.get_function("tensor_matrix")
        gpu_tensorM(gpu_out, gpu_A, gpu_size, grid=grid, block=block)
        return gpu_out.get()

    @staticmethod
    def vector_permutation(A, mapping, inplace):
        mapping = np.array(mapping, dtype=np.int64)

        if not A.shape[0] == 1 << mapping.shape[0]:
            raise IndexError("Indices do not match!")

        idx_mapping = mapping_augment(mapping)
        gpu_idx = gpuarray.to_gpu(idx_mapping)
        gpu_A = gpuarray.to_gpu(A)

        if inplace:
            A = gpuarray.take(gpu_A, gpu_idx).get()
        
        return gpuarray.take(gpu_A, gpu_idx).get()

    @staticmethod
    def matrix_permutation(A, mapping, inplace):
        mapping = np.array(mapping, dtype=np.int64)

        if not A.shape[0] == 1 << mapping.shape[0]:
            raise IndexError("Indices do not match!")

        idx_mapping = mapping_augment(mapping)
        gpu_idx = gpuarray.to_gpu(idx_mapping)
        gpu_A = gpuarray.to_gpu(A[:,idx_mapping])

        if inplace:
            A = gpuarray.take(gpu_A, gpu_idx).get()
        
        return gpuarray.take(gpu_A, gpu_idx).get()
