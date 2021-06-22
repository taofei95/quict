#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午3:34
# @Author  : Han Yu
# @File    : gpu_constant_calculator_refine

import numpy as np
import cupy as cp

from QuICT.core import *


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


def gate_dot_vector_single_precision(
    gate : BasicGate,
    mat,
    vec,
    vec_bit
):
    if gate.type() == GATE_ID["H"]:
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block
        HGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (vec_bit - 1 - gate.targ, mat, vec)
        )
    elif gate.type() == GATE_ID["CRz"]:
        cindex = vec_bit - 1 - gate.carg
        tindex = vec_bit - 1 - gate.targ

        task_number = 1 << (vec_bit - 2)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (cindex, tindex, mat, vec)
        )
    else:
        raise Exception("ss")

    cp.cuda.Device().synchronize()


def gate_dot_vector_double_precision(
    gate : BasicGate,
    mat,
    vec,
    vec_bit
):
    if gate.type() == GATE_ID["H"]:
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block
        HGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (vec_bit - 1 - gate.targ, mat, vec)
        )
    elif gate.type() == GATE_ID["CRz"]:
        cindex = vec_bit - 1 - gate.carg
        tindex = vec_bit - 1 - gate.targ

        task_number = 1 << (vec_bit - 2)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (cindex, tindex, mat, vec)
        )
    else:
        raise Exception("Unsupported gate.")

    cp.cuda.Device().synchronize()


def gate_kernel_by_precision(precision):
    if precision == np.complex64:
        return gate_dot_vector_single_precision
    else:
        return gate_dot_vector_double_precision
