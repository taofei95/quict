#!/usr/bin/env python
# -*- coding:utf8 -*-

import numpy as np
import cupy as cp


HGate_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void HGateSingle(int index, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);

        complex<float> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2] + vec[_1]*mat[3];
    }
    ''', 'HGateSingle')


CRZGate_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateSingle(int cindex, int tindex, const complex<float>* mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int gw=0, _0=0;

        if(tindex > cindex){
            gw = label >> cindex << (cindex + 1);
            _0 = (1 << cindex) + (gw & ((1 << tindex) - (1 << cindex))) + (gw >> tindex << (tindex + 1)) + (label & ((1 << cindex) - 1));
        }
        else
        {
            gw = label >> tindex << (tindex + 1);
            _0 = (1 << cindex) + (gw & ((1 << cindex) - (1 << tindex))) + (gw >> cindex << (cindex + 1)) + (label & ((1 << tindex) - 1));
        }

        int _1 = _0 + (1 << tindex);

        vec[_0] = vec[_0]*mat[10];
        vec[_1] = vec[_1]*mat[15];
    }
    ''', 'CRZGateSingle')


HGate_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void HGateDouble(int index, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);

        complex<double> temp_0 = vec[_0];
        vec[_0] = vec[_0]*mat[0] + vec[_1]*mat[1];
        vec[_1] = temp_0*mat[2] + vec[_1]*mat[3];
    }
    ''', 'HGateDouble')


CRZGate_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateDouble(int cindex, int tindex, const complex<double>* mat, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int gw=0, _0=0;

        if(tindex > cindex){
            gw = label >> cindex << (cindex + 1);
            _0 = (1 << cindex) + (gw & ((1 << tindex) - (1 << cindex))) + (gw >> tindex << (tindex + 1)) + (label & ((1 << cindex) - 1));
        }
        else
        {
            gw = label >> tindex << (tindex + 1);
            _0 = (1 << cindex) + (gw & ((1 << cindex) - (1 << tindex))) + (gw >> cindex << (cindex + 1)) + (label & ((1 << tindex) - 1));
        }

        int _1 = _0 + (1 << tindex);

        vec[_0] = vec[_0]*mat[10];
        vec[_1] = vec[_1]*mat[15];
    }
    ''', 'CRZGateDouble')


def HGate_matrixdot(t_index, mat, vec, vec_bit, sync: bool = False):
    """
    HGate dot function.
    """
    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block
    if vec.dtype == np.complex64:
        HGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    elif vec.dtype == np.complex128:
        HGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )
    else:
        raise TypeError(f"Unsupported type of {vec.dtype}.")

    if sync:
        cp.cuda.Device().synchronize()


def CRzGate_matrixdot(c_index, t_index, mat, vec, vec_bit, sync: bool = False):
    """
    CRzGate dot function.
    """
    task_number = 1 << (vec_bit - 2)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        CRZGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (c_index, t_index, mat, vec)
        )
    elif vec.dtype == np.complex128:
        CRZGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (c_index, t_index, mat, vec)
        )
    else:
        raise TypeError(f"Unsupported type of {vec.dtype}.")

    if sync:
        cp.cuda.Device().synchronize()

