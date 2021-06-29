#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午3:34
# @Author  : Han Yu
# @File    : gpu_constant_calculator_refine

import numpy as np
import cupy as cp

from QuICT.core import *


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


class GateFuncD:
    """
    The class is the aggregation of gate dot functions with double precision.
    """
    @classmethod
    def HGate_matrixdot(self, t_index, mat, vec, vec_bit, sync: bool = False):
        """
        HGate dot function with double precision.
        """
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block
        HGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot(self, c_index, t_index, mat, vec, vec_bit, sync: bool = False):
        """
        CRzGate dot function with double precision.
        """
        task_number = 1 << (vec_bit - 2)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_double_kernel(
            (block_num,),
            (thread_per_block,),
            (c_index, t_index, mat, vec)
            )

        if sync:
            cp.cuda.Device().synchronize()


class GateFuncMD:
    """
    The class is the aggregation of gate dot double precision functions which using 
    in the multi-GPUs.
    """
    @classmethod
    def CRzGate_matrixdot_pb(self, _0_1, mat, vec, vec_bit, sync: bool = True):
        """ 
        Special CRzGate dot function for the multi-GPUs with double precision. 
        Using when both c_index and t_index are higher than the maximum qubits 
        in the current device.
        """
        mat_value = mat[10] if _0_1 else mat[15]

        task_number = 1 << vec_bit
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_dd(
            (block_num,),
            (thread_per_block,),
            (mat_value, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot_pc(self, _0_1, c_index, mat, vec, vec_bit, sync: bool = True):
        """ 
        Special CRzGate dot function for the multi-GPUs with double precision. 
        Using when the t_index is higher than the maximum qubits in the current
        device, and the c_index doesn't.
        """
        mat_value = mat[10] if _0_1 else mat[15]

        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_dc(
            (block_num,),
            (thread_per_block,),
            (c_index, mat_value, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot_pt(self, t_index, mat, vec, vec_bit, sync: bool = True):
        """ 
        Special CRzGate dot function for the multi-GPUs with double precision.
        Using when the c_index is higher than the maximum qubits in the current
        device, and the t_index doesn't.
        """
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_dt(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()
