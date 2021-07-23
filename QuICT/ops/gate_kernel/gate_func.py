#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/7/5 上午10:43
# @Author  : Kaiqi Li
# @File    : gate_func

import cupy as cp
import numpy as np

import random


__outward_functions = ["HGate_matrixdot", "CRzGate_matrixdot", "MeasureGate_Measure"]


prop_add = cp.ElementwiseKernel(
    'T x, raw T y, int32 index', 'T z',
    'z = (i & index) ? 0 : abs(x) * abs(x)',
    'prop_add')


MeasureGate_prop_kernel = cp.ReductionKernel(
    'T x',
    'T y',
    'x',
    'a + b',
    'y = abs(a)',
    '0',
    'MeasureGate_prop_kernel')


MeasureGate0_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate0Single(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] * generation;
        vec[_0 + (1 << index)] = complex<float>(0, 0);
    }
    ''', 'MeasureGate0Single')


MeasureGate1_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate1Single(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);
        vec[_0] = complex<float>(0, 0);
        vec[_1] = vec[_1] * generation;
    }
    ''', 'MeasureGate1Single')


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


MeasureGate0_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate0Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] * generation;
        vec[_0 + (1 << index)] = complex<double>(0, 0);
    }
    ''', 'MeasureGate0Double')


MeasureGate1_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void MeasureGate1Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        int _1 = _0 + (1 << index);
        vec[_0] = complex<double>(0, 0);
        vec[_1] = vec[_1] * generation;
    }
    ''', 'MeasureGate1Double')


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


def MeasureGate_Measure(index, vec, vec_bit, sync: bool = False):
    """
    Measure Gate Measure.
    """
    prob = prop_add(vec, vec, 1 << index)
    prob = MeasureGate_prop_kernel(prob, axis = 0).real
    _0 = random.random()
    _1 = _0 > prob
    prob = prob.get()

    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if not _1:
        if vec.dtype == np.complex64:
            alpha = np.float32(1 / np.sqrt(prob))
            MeasureGate0_single_kernel(
                (block_num, ),
                (thread_per_block, ),
                (index, alpha, vec)
            )
        else:
            alpha = np.float64(1 / np.sqrt(prob))
            MeasureGate0_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
    else:
        if vec.dtype == np.complex64:
            alpha = np.float32(1 / np.sqrt(1 - prob))
            MeasureGate1_single_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
        else:
            alpha = np.float64(1 / np.sqrt(1 - prob))
            MeasureGate1_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
    if sync:
        cp.cuda.Device().synchronize()

    return _1
