#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/29 下午5:35
# @Author  : Kaiqi Li
# @File    : gpu_calculator

import math
import numpy as np
import cupy as cp

from typing import *


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


def dot(A, B, gpu_out: bool = True, sync: bool = True):
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

    gpu_result = cp.dot(gpu_A, gpu_B)

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


tensor_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void tensorsingle(complex<float>* x, complex<float>* y, complex<float>* out, int cx, int ry, int cy) {
        int out_id = blockDim.x * blockIdx.x + threadIdx.x;
        int out_width = cx * cy;
        int y_size = ry * cy;
        int x_id = (out_id/out_width)/ry * cx + (out_id%out_width) / cy;
        int y_id = (out_id/out_width)%ry * cy + (out_id%out_width) % cy;
        out[out_id] = x[x_id] * y[y_id];
    }
    ''', 'tensorsingle')


tensor_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void tensordouble(complex<double>* x, complex<double>* y, complex<double>* out, int cx, int ry, int cy) {
        int out_id = blockDim.x * blockIdx.x + threadIdx.x;
        int out_width = cx * cy;
        int y_size = ry * cy;
        int x_id = (out_id/out_width)/ry * cx + (out_id%out_width) / cy;
        int y_id = (out_id/out_width)%ry * cy + (out_id%out_width) % cy;
        out[out_id] = x[x_id] * y[y_id];
    }
    ''', 'tensordouble')


def tensor(A, B, gpu_out: bool = True, sync: bool = True):
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

    precision = A.dtype

    gpu_result = cp.empty((gpu_A.shape[0]*gpu_B.shape[0], gpu_A.shape[1]*gpu_B.shape[1]), dtype=precision)
    core_number = gpu_result.size

    if precision == np.complex64:
        tensor_single_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_B, gpu_result, cp.int32(gpu_A.shape[1]), cp.int32(gpu_B.shape[0]), cp.int32(gpu_B.shape[1]))
        )
    else:
        tensor_double_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_B, gpu_result, cp.int32(gpu_A.shape[1]), cp.int32(gpu_B.shape[0]), cp.int32(gpu_B.shape[1]))
        )

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


matrixt_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_tensorI_single(const complex<float>* A, complex<float>* out, int n, int m, int rx, int cx) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int x_id = tid/cx;
        int y_id = (x_id/(rx*m))*(m*cx) + (tid%cx)*m + (x_id%(rx*m))%m;
        int out_xid = (x_id%(rx*m))/m;
        int out_yid = tid%cx;
        out[x_id*(cx*n*m) + y_id] = A[out_xid*cx + out_yid];
    }
    ''', 'matrix_tensorI_single')


matrixt_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_tensorI_double(const complex<double>* A, complex<double>* out, int n, int m, int rx, int cx) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int x_id = tid/cx;
        int y_id = (x_id/(rx*m))*(m*cx) + (tid%cx)*m + (x_id%(rx*m))%m;
        int out_xid = (x_id%(rx*m))/m;
        int out_yid = tid%cx;
        out[x_id*(cx*n*m) + y_id] = A[out_xid*cx + out_yid];
    }
    ''', 'matrix_tensorI_double')


def MatrixTensorI(A, n, m, gpu_out: bool = True, sync: bool = True):
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
    precision = A.dtype
    gpu_A = cp.array(A) if type(A) is np.ndarray else A

    gpu_result = cp.zeros((row_a*n*m, col_a*n*m), dtype=precision)
    core_number = gpu_A.size*n*m
    if precision == np.complex64:
        matrixt_single_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, cp.int32(n), cp.int32(m), cp.int32(row_a), cp.int32(col_a))
        )
    else:
        matrixt_double_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, cp.int32(n), cp.int32(m), cp.int32(row_a), cp.int32(col_a))
        )

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


vectorp_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void vector_single_permutation(const complex<float>* x, complex<float>* y, int* mapping, int m) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xid = 0;
        int l = 0;
        for(int i = 0; i < m; i++){
            if ((i != m - 1) && ((mapping[i] + 1) == mapping[i+1])){
                l += 1;
            }else{
                xid |= ((tid >> (m - 1 - mapping[i])) & ((1 << (l + 1)) - 1)) << (m - 1 - i);
                l = 0;
            }
        }
        y[tid] = x[xid];
    }
    ''', 'vector_single_permutation')


vectorp_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void vector_double_permutation(const complex<double>* x, complex<double>* y, int* mapping, int m) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xid = 0;
        int l = 0;
        for(int i = 0; i < m; i++){
            if ((i != m - 1) && ((mapping[i] + 1) == mapping[i+1])){
                l += 1;
            }else{
                xid |= ((tid >> (m - 1 - mapping[i])) & ((1 << (l + 1)) - 1)) << (m - 1 - i);
                l = 0;
            }
        }
        y[tid] = x[xid];
    }
    ''', 'vector_double_permutation')


def VectorPermutation(A, mapping, changeInput: bool = False, gpu_out: bool = True, sync: bool = True):
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

    if mapping.dtype == np.int64:
        mapping = mapping.astype(np.int32)

    # data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_mapping = cp.array(mapping)
    gpu_result = cp.empty_like(gpu_A)
    core_number = gpu_result.size

    if A.dtype == np.complex64:
        vectorp_single_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping, cp.int32(n))
        )
    else:
        vectorp_double_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping, cp.int32(n))
        )

    if changeInput:
        A[:] = gpu_result.get() if type(A) is np.ndarray else gpu_result

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


matrixp_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_single_permutation(const complex<float>* x, complex<float>* y, int* mapping, int m) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int len = 1 << m;
        int rx = tid/len;
        int cx = tid%len;
        int rtemp = 0;
        int ctemp = 0;
        int l = 1;
        for(int i = 0; i < m; i++){
            if ((i != m - 1) && ((mapping[i] | 1) == mapping[i+1])){
                l = (l << 1) | 1;
            }else{
                rtemp |= ((rx >> (m - 1 - mapping[i])) & l) << (m - 1 - i);
                ctemp |= ((cx >> (m - 1 - mapping[i])) & l) << (m - 1 - i);
                l = 1;
            }
        }
        y[tid] = x[rtemp*len + ctemp];
    }
    ''', 'matrix_single_permutation')


matrixp_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_double_permutation(const complex<double>* x, complex<double>* y, int* mapping, int m) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int len = 1 << m;
        int rx = tid/len;
        int cx = tid%len;
        int rtemp = 0;
        int ctemp = 0;
        int l = 1;
        for(int i = 0; i < m; i++){
            if ((i != m - 1) && ((mapping[i] | 1) == mapping[i+1])){
                l = (l << 1) | 1;
            }else{
                rtemp |= ((rx >> (m - 1 - mapping[i])) & l) << (m - 1 - i);
                ctemp |= ((cx >> (m - 1 - mapping[i])) & l) << (m - 1 - i);
                l = 1;
            }
        }
        y[tid] = x[rtemp*len + ctemp];
    }
    ''', 'matrix_double_permutation')


def MatrixPermutation(A, mapping, changeInput: bool = False, gpu_out: bool = True, sync: bool = True):
    """ permute mat with mapping, inplace

    Args:
        A(np.array<np.complex>): the matrix A.
        mapping(np.array<int>): the qubit mapping.
        changeInput(bool): whether changes in A.
        gpu_out(bool): return result from GPU.
    """
    row_a, n = A.shape[0], mapping.shape[0]
    if not row_a == 1 << n:
        raise IndexError("Indices do not match!")

    if mapping.dtype == np.int64:
        mapping = mapping.astype(np.int32)

    # data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_mapping = cp.array(mapping)
    gpu_result = cp.empty_like(gpu_A)

    core_number = gpu_result.size
    if A.dtype == np.complex64:
        matrixp_single_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping, cp.int32(n))
        )
    else:
        matrixp_double_kernel(
            (math.ceil(core_number/1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping, cp.int32(n))
        )

    if changeInput:
        A[:, :] = gpu_result.get() if type(A) is np.ndarray else gpu_result

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


matrix_dot_vector_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_dot_vector_single(const complex<float>* mat, int mat_bit, int mat_len, complex<float>* vec, int* affect_args, int* aff_argsorts, complex<float>* anc) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int other = tid & ((1 << aff_argsorts[0]) - 1);
        int gw = tid >> aff_argsorts[0] << (aff_argsorts[0] + 1);
        for(int i = 1; i < mat_bit; i++){
            other += gw & ((1 << aff_argsorts[i]) - (1 << aff_argsorts[i - 1]));
            gw = gw >> aff_argsorts[i] << (aff_argsorts[i] + 1);
        }
        other += gw;

        for(int i = 0; i < mat_len; i++){
            int now = other;
            for(int k = 0; k < mat_bit; k++){
                if (i & (1 << k)){
                    now += 1 << affect_args[mat_bit - 1 - k];
                }
            }
            anc[now] = 0;
            for(int k = 0; k < mat_len; k++){
                int shift = other;
                for(int l = 0; l < mat_bit; l++){
                    if (k & (1 << l)){
                        shift += 1 << affect_args[mat_bit - 1 - l];
                    }
                }
                anc[now] += mat[i*mat_len + k] * vec[shift];
            }
        }

    }
    ''', 'matrix_dot_vector_single')


matrix_dot_vector_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void matrix_dot_vector_double(const complex<double>* mat, int mat_bit, int mat_len, complex<double>* vec, int* affect_args, int* aff_argsorts, complex<double>* anc) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int other = tid & ((1 << aff_argsorts[0]) - 1);
        int gw = tid >> aff_argsorts[0] << (aff_argsorts[0] + 1);
        for(int i = 1; i < mat_bit; i++){
            other += gw & ((1 << aff_argsorts[i]) - (1 << aff_argsorts[i - 1]));
            gw = gw >> aff_argsorts[i] << (aff_argsorts[i] + 1);
        }
        other += gw;

        for(int i = 0; i < mat_len; i++){
            int now = other;
            for(int k = 0; k < mat_bit; k++){
                if (i & (1 << k)){
                    now += 1 << affect_args[mat_bit - 1 - k];
                }
            }
            anc[now] = 0;
            for(int k = 0; k < mat_len; k++){
                int shift = other;
                for(int l = 0; l < mat_bit; l++){
                    if (k & (1 << l)){
                        shift += 1 << affect_args[mat_bit - 1 - l];
                    }
                }
                anc[now] += mat[i*mat_len + k] * vec[shift];
            }
        }
        
    }
    ''', 'matrix_dot_vector_double')


def matrix_dot_vector(
    mat,
    mat_bit,
    vec,
    vec_bit,
    affect_args,
    auxiliary_vec,
    sync
):
    mat_length = 1 << mat_bit

    task_number = 1 << (vec_bit - mat_bit)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    for i in range(mat_bit):
       affect_args[i] = vec_bit - 1 - affect_args[i]

    affect_args_sorts = affect_args.copy()
    affect_args_sorts.sort()

    mat = cp.array(mat, dtype=vec.dtype)
    mat_bit = np.int32(mat_bit)
    mat_length = np.int64(mat_length)
    affect_args = htod(np.array(affect_args, dtype=np.int32))
    affect_args_sorts = htod(np.array(affect_args_sorts, dtype=np.int32))
    
    if vec.dtype == np.complex64:
        matrix_dot_vector_single_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, mat_bit, mat_length, vec, affect_args, affect_args_sorts, auxiliary_vec)
        )
    else:
        matrix_dot_vector_double_kernel(
            (block_num,),
            (thread_per_block,),
            (mat, mat_bit, mat_length, vec, affect_args, affect_args_sorts, auxiliary_vec)
        )

    if sync:
        cp.cuda.Device().synchronize()
