#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
import cupy as cp


__outward_functions = ["CRzGate_matrixdot_pb", "CRzGate_matrixdot_pc", "CRzGate_matrixdot_pt"]


CRZGate_kernel_special_sd = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateSingleSD(const complex<float> mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*mat;
    }
    ''', 'CRZGateSingleSD')


CRZGate_kernel_special_sc = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateSingleSC(int cindex, const complex<float> mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int _0 = (1 << cindex) + (label & ((1 << cindex) - 1)) + (label >> cindex << (cindex + 1));

        vec[_0] = vec[_0]*mat;
    } 
    ''', 'CRZGateSingleSC')


CRZGate_kernel_special_st = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateSingleST(int tindex, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int _0 = (label & ((1 << tindex) - 1)) + (label >> tindex << (tindex + 1));
        int _1 = _0 + (1 << tindex);

        vec[_0] = vec[_0]*mat[10];
        vec[_1] = vec[_1]*mat[15];
    }
    ''', 'CRZGateSingleST')


CRZGate_kernel_special_dd = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateDoubleSD(const complex<double> mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*mat;
    }
    ''', 'CRZGateDoubleSD')


CRZGate_kernel_special_dc = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateDoubleSC(int cindex, const complex<double> mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int _0 = (1 << cindex) + (label & ((1 << cindex) - 1)) + (label >> cindex << (cindex + 1));

        vec[_0] = vec[_0]*mat;
    } 
    ''', 'CRZGateDoubleSC')


CRZGate_kernel_special_dt = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateDoubleST(int tindex, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int _0 = (label & ((1 << tindex) - 1)) + (label >> tindex << (tindex + 1));
        int _1 = _0 + (1 << tindex);

        vec[_0] = vec[_0]*mat[10];
        vec[_1] = vec[_1]*mat[15];
    }
    ''', 'CRZGateDoubleST')


def CRzGate_matrixdot_pb(_0_1, mat, vec, vec_bit, sync: bool = True):
    """ 
    Special CRzGate dot function for the multi-GPUs. Using when both c_index 
    and t_index are higher than the maximum qubits in the current device.
    """
    mat_value = mat[10] if _0_1 else mat[15]

    task_number = 1 << vec_bit
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        CRZGate_kernel_special_sd(
            (block_num,),
            (thread_per_block,),
            (mat_value, vec)
        )
    elif vec.dtype == np.complex128:
        CRZGate_kernel_special_dd(
            (block_num,),
            (thread_per_block,),
            (mat_value, vec)
        )
    else:
        raise TypeError(f"Unsupported type of {vec.dtype}.")

    if sync:
        cp.cuda.Device().synchronize()


def CRzGate_matrixdot_pc(_0_1, c_index, mat, vec, vec_bit, sync: bool = True):
    """ 
    Special CRzGate dot function for the multi-GPUs. Using when the t_index is higher 
    than the maximum qubits in the current device, and the c_index doesn't.
    """
    mat_value = mat[10] if _0_1 else mat[15]

    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        CRZGate_kernel_special_sc(
            (block_num,),
            (thread_per_block,),
            (c_index, mat_value, vec)
        )
    elif vec.dtype == np.complex128:
        CRZGate_kernel_special_dc(
            (block_num,),
            (thread_per_block,),
            (c_index, mat_value, vec)
        )
    else:
        raise TypeError(f"Unsupported type of {vec.dtype}.")

    if sync:
        cp.cuda.Device().synchronize()


def CRzGate_matrixdot_pt(t_index, mat, vec, vec_bit, sync: bool = True):
    """ 
    Special CRzGate dot function for the multi-GPUs. Using when the c_index is 
    higher than the maximum qubits in the current device, and the t_index doesn't.
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block


    if vec.dtype == np.complex64:
        CRZGate_kernel_special_st(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    elif vec.dtype == np.complex128:
        CRZGate_kernel_special_dt(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        raise TypeError(f"Unsupported type of {vec.dtype}.")

    if sync:
        cp.cuda.Device().synchronize()
