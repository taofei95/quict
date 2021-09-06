#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/7/5 上午10:43
# @Author  : Kaiqi Li
# @File    : gate_func

import cupy as cp
import numpy as np

import random


__outward_functions = ["MeasureGate_Apply", "ResetGate_Apply", "PermGate_Apply"]


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


ResetGate0_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate0Float(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] / generation;
        vec[_0 + (1 << index)] = complex<float>(0, 0);
    }
    ''', 'ResetGate0Float')


ResetGate1_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate1Float(const int index, const float generation, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        int _1 = _0 + (1 << index)

        vec[_0] = vec[_1];
        vec[_1] = complex<float>(0, 0);
    }
    ''', 'ResetGate1Float')


ResetGate0_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate0Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _0 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1));
        vec[_0] = vec[_0] / generation;
        vec[_0 + (1 << index)] = complex<double>(0, 0);
    }
    ''', 'ResetGate0Double')


ResetGate1_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void ResetGate1Double(const int index, const double generation, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;
        int _1 = (label & ((1 << index) - 1)) 
                + (label >> index << (index + 1))
                + (1 << index);

        vec[label] = vec[_1];
    }
    ''', 'ResetGate1Double')


PermGate_single_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PermGate(const int idx_len, int vec_bit, int* indexes, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        complex<float> temp[1 << 12];
        int swap_idx=0, vec_idx=0;
        for(int i = 0; i < idx_len; i++){
            swap_idx = indexes[i];
            if (swap_idx != i){
                vec_idx = (i << vec_bit) + label;
                temp[i] = vec[vec_idx];
                if (swap_idx < i){
                    vec[vec_idx] = temp[swap_idx];
                }else{
                    vec[vec_idx] = vec[(swap_idx << vec_bit) + label];
                }
            }
        }
    }
    ''', 'PermGate')


PermGate_double_kernel = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void PermGate(const int idx_len, int vec_bit, int* indexes, complex<double>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        complex<double> temp[1 << 12];
        int swap_idx=0, vec_idx=0;
        for(int i = 0; i < idx_len; i++){
            swap_idx = indexes[i];
            if (swap_idx != i){
                vec_idx = (i << vec_bit) + label;
                temp[i] = vec[vec_idx];
                if (swap_idx < i){
                    vec[vec_idx] = temp[swap_idx];
                }else{
                    vec[vec_idx] = vec[(swap_idx << vec_bit) + label];
                }
            }
        }
    }
    ''', 'PermGate')


def MeasureGate_Apply(index, vec, vec_bit, sync: bool = False, multigpu_prob = None):
    """
    Measure Gate Measure.
    """
    if not multigpu_prob:
        prob = prop_add(vec, vec, 1 << index)
        prob = MeasureGate_prop_kernel(prob, axis = 0).real
        prob = prob.get()
    else:
        prob = multigpu_prob

    _0 = random.random()
    _1 = _0 > prob

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


def ResetGate_Apply(index, vec, vec_bit, sync: bool = False, multigpu_prob = None):
    """
    Measure Gate Measure.
    """
    if not multigpu_prob:
        prob = prop_add(vec, vec, 1 << index)
        prob = MeasureGate_prop_kernel(prob, axis = 0).real
        prob = prob.get()
    else:
        prob = multigpu_prob

    task_number = 1 << (vec_bit - 1)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    alpha = np.float64(np.sqrt(prob))

    if alpha < 1e-6:
        if vec.dtype == np.complex64:
            ResetGate1_single_kernel(
                (block_num, ),
                (thread_per_block,),
                (index, alpha, vec)
            )
        else:
            ResetGate1_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
    else:
        if vec.dtype == np.complex64:
            ResetGate0_single_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )
        else:
            ResetGate0_double_kernel(
                (block_num,),
                (thread_per_block,),
                (index, alpha, vec)
            )

    if sync:
        cp.cuda.Device().synchronize()


def PermGate_Apply(indexes, vec, vec_bit, sync: bool = False):
    len_indexes = len(indexes)
    targets = np.int32(np.log2(len(indexes)))
    indexes = cp.array(indexes, dtype=np.int32)

    task_number = 1 << (vec_bit - targets)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    if vec.dtype == np.complex64:
        PermGate_single_kernel(
            (block_num, ),
            (thread_per_block,),
            (len_indexes, vec_bit - targets, indexes, vec)
        )
    else:
        PermGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (len_indexes, vec_bit - targets, indexes, vec)
        )

    if sync:
        cp.cuda.Device().synchronize()
