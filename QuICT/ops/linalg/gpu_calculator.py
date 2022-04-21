#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/29 下午5:35
# @Author  : Kaiqi Li
# @File    : gpu_calculator

import math
import numpy as np
import cupy as cp


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

    if A.ndim == 1:
        row_a, col_a = A.shape[0], 1
    else:
        row_a, col_a = A.shape

    if B.ndim == 1:
        row_b, col_b = B.shape[0], 1
    else:
        row_b, col_b = B.shape

    gpu_result = cp.empty((row_a * row_b, col_a * col_b), dtype=precision)
    core_number = gpu_result.size

    if precision == np.complex64:
        tensor_single_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_B, gpu_result, cp.int32(col_a), cp.int32(row_b), cp.int32(col_b))
        )
    else:
        tensor_double_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_B, gpu_result, cp.int32(col_a), cp.int32(row_b), cp.int32(col_b))
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

    gpu_result = cp.zeros((row_a * n * m, col_a * n * m), dtype=precision)
    core_number = gpu_A.size * n * m
    if precision == np.complex64:
        matrixt_single_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, cp.int32(n), cp.int32(m), cp.int32(row_a), cp.int32(col_a))
        )
    else:
        matrixt_double_kernel(
            (math.ceil(core_number / 1024),),
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
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping, cp.int32(n))
        )
    else:
        vectorp_double_kernel(
            (math.ceil(core_number / 1024),),
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
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping, cp.int32(n))
        )
    else:
        matrixp_double_kernel(
            (math.ceil(core_number / 1024),),
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
    void matrix_dot_vector_single(
        const complex<float>* mat,
        int mat_bit,
        int mat_len,
        complex<float>* vec,
        int* affect_args,
        int* aff_argsorts,
        complex<float>* anc
    ){
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
    void matrix_dot_vector_double(
        const complex<double>* mat,
        int mat_bit,
        int mat_len,
        complex<double>* vec,
        int* affect_args,
        int* aff_argsorts,
        complex<double>* anc
    ){
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
    sync: bool = True
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
    affect_args = cp.array(affect_args, dtype=np.int32)
    affect_args_sorts = cp.array(affect_args_sorts, dtype=np.int32)

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


simplevp_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void simple_vp_single(const complex<float>* x, complex<float>* y, int* mapping) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xid = mapping[tid];
        y[tid] = x[xid];
    }
    ''', 'simple_vp_single')


simplecp_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void simple_vp_double(const complex<double>* x, complex<double>* y, int* mapping) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int xid = mapping[tid];
        y[tid] = x[xid];
    }
    ''', 'simple_vp_double')


def simple_vp(
    A,
    mapping,
    changeInput: bool = False,
    gpu_out: bool = True,
    sync: bool = True
):
    """ permutaion A with mapping, inplace

    Args:
        A(np.array<np.complex>): the vector A.
        mapping(np.array<int>): the mapping.
        changeInput(bool): whether changes in A.
        gpu_out(bool): return result from GPU.

    Returns:
        np.array<np.complex>: the result of Permutation
    """
    row_a, n = A.shape[0], mapping.shape[0]
    if row_a != n:
        raise IndexError("Indices do not match!")

    if mapping.dtype == np.int64:
        mapping = mapping.astype(np.int32)

    # data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_mapping = cp.array(mapping)
    gpu_result = cp.empty_like(gpu_A)
    core_number = gpu_result.size

    if A.dtype == np.complex64:
        simplevp_single_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping)
        )
    else:
        simplecp_double_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_mapping)
        )

    if changeInput:
        A[:] = gpu_result.get() if type(A) is np.ndarray else gpu_result

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


qubit_vp_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void qubit_vp_single(
        const complex<float>* x,
        complex<float>* y,
        int* qargs,
        int* mapping,
        const int n,
        const int N
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        for (int i = 0; i < n; i++){
            int block_swap_idx = mapping[i];
            int ori_idx=0, swap_idx=0;
            for (int j = 0; j < N; j++){
                int label = N - 1 - j;
                if (qargs[j] < 0){
                    int temp_tvalue = (tid & (1 << (-qargs[j] - 1))) << (label - (-qargs[j] - 1));
                    ori_idx += temp_tvalue;
                    swap_idx += temp_tvalue;
                }else{
                    ori_idx += (i & (1 << qargs[j])) << (label - qargs[j]);
                    swap_idx += (block_swap_idx & (1 << qargs[j])) << (label - qargs[j]);
                }
            }

            y[ori_idx] = x[swap_idx];
        }
    }
    ''', 'qubit_vp_single')


qubit_vp_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void qubit_vp_double(
        const complex<double>* x,
        complex<double>* y,
        int* qargs,
        int* mapping,
        const int n,
        const int N
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        for (int i = 0; i < n; i++){
            int block_swap_idx = mapping[i];
            int ori_idx=0, swap_idx=0;
            for (int j = 0; j < N; j++){
                int label = N - 1 - j;
                if (qargs[j] < 0){
                    int temp_tvalue = (tid & (1 << (-qargs[j] - 1))) << (label - (-qargs[j] - 1));
                    ori_idx += temp_tvalue;
                    swap_idx += temp_tvalue;
                }else{
                    ori_idx += (i & (1 << qargs[j])) << (label - qargs[j]);
                    swap_idx += (block_swap_idx & (1 << qargs[j])) << (label - qargs[j]);
                }
            }
            y[ori_idx] = x[swap_idx];
        }
    }
    ''', 'qubit_vp_double')


def qubit_vp(
    A,
    qargs,
    mapping,
    changeInput: bool = False,
    gpu_out: bool = True,
    sync: bool = True
):
    """ permutaion A with mapping by given qubits index, inplace

    Args:
        A(np.array<np.complex>): the vector A.
        qargs (np.array<int>): the target qubits' index
        mapping(np.array<int>): the mapping.
        changeInput(bool): whether changes in A.
        gpu_out(bool): return result from GPU.

    Returns:
        np.array<np.complex>: the result of Permutation
    """
    N = int(np.log2(A.size))
    qubits, n = len(qargs), mapping.shape[0]
    if 1 << qubits != n:
        raise IndexError("Indices do not match!")

    if mapping.dtype == np.int64:
        mapping = mapping.astype(np.int32)

    if qargs.dtype == np.int64:
        qargs = qargs.astype(np.int32)

    qargs = _qargs_fill(qargs, N)

    # data in GPU
    gpu_A = cp.array(A) if type(A) is np.ndarray else A
    gpu_result = cp.empty_like(gpu_A)

    gpu_mapping = cp.array(mapping)
    gpu_qargs = cp.array(qargs)
    gpu_n = cp.int32(n)
    gpu_N = cp.int32(N)
    core_number = gpu_result.size // gpu_mapping.size

    if A.dtype == np.complex64:
        qubit_vp_single_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_qargs, gpu_mapping, gpu_n, gpu_N)
        )
    else:
        qubit_vp_double_kernel(
            (math.ceil(core_number / 1024),),
            (min(1024, core_number),),
            (gpu_A, gpu_result, gpu_qargs, gpu_mapping, gpu_n, gpu_N)
        )

    if changeInput:
        A[:] = gpu_result.get() if type(A) is np.ndarray else gpu_result

    if sync:
        cp.cuda.Device().synchronize()

    if gpu_out:
        return gpu_result.get()

    return gpu_result


def _qargs_fill(qargs, qubits):
    result = np.zeros(qubits, dtype=np.int32)

    count = -(qubits - len(qargs))
    for i in range(qubits):
        if i not in qargs:
            result[i] = count
            count += 1
        else:
            result[i] = len(qargs) - 1 - np.argwhere(qargs == i)

    return result


kernel_funcs = list(locals().keys())
for name in kernel_funcs:
    if name.endswith("kernel"):
        locals()[name].compile()
