#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午3:34
# @Author  : Han Yu
# @File    : gpu_constant_calculator_refine

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

CRZGate_kernel_special_d = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateSingleSD(const complex<float> mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        vec[label] = vec[label]*mat;
    }
    ''', 'CRZGateSingleSD')

CRZGate_kernel_special_c = cp.RawKernel(r'''
    #include <cupy/complex.cuh>
    extern "C" __global__
    void CRZGateSingleSC(int cindex, const complex<float> mat, complex<float>* vec) {
        int label = blockDim.x * blockIdx.x + threadIdx.x;

        int _0 = (1 << cindex) + (label & ((1 << cindex) - 1)) + (label >> cindex << (cindex + 1));

        vec[_0] = vec[_0]*mat;
    } 
    ''', 'CRZGateSingleSC')

CRZGate_kernel_special_t = cp.RawKernel(r'''
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


class GateFuncS:
    """
    The class is the aggregation of gate dot functions with single precision.
    """
    @classmethod
    def HGate_matrixdot(self, t_index, mat, vec, vec_bit, sync: bool = True):
        """
        HGate dot function with single precision.
        """
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block
        HGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot(self, c_index, t_index, mat, vec, vec_bit, sync: bool = True):
        """
        CRzGate dot function with single precision.
        """
        task_number = 1 << (vec_bit - 2)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (c_index, t_index, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()


class GateFuncMS:
    """
    The class is the aggregation of gate dot single precision functions which using 
    in the multi-GPUs.
    """
    @classmethod
    def CRzGate_matrixdot_pd(self, _0_1, mat, vec, vec_bit, sync: bool = True):
        """ 
        Special CRzGate dot function for the multi-GPUs with single precision. 
        Using when both c_index and t_index are higher than the maximum qubits 
        in the current device.
        """
        mat_value = mat[10] if _0_1 else mat[15]

        task_number = 1 << vec_bit
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_d(
            (block_num,),
            (thread_per_block,),
            (mat_value, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot_pc(self, _0_1, c_index, mat, vec, vec_bit, sync: bool = True):
        """ 
        Special CRzGate dot function for the multi-GPUs with single precision. 
        Using when the t_index is higher than the maximum qubits in the current
        device, and the c_index doesn't.
        """
        mat_value = mat[10] if _0_1 else mat[15]

        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_c(
            (block_num,),
            (thread_per_block,),
            (c_index, mat_value, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot_pt(self, t_index, mat, vec, vec_bit, sync: bool = True):
        """ 
        Special CRzGate dot function for the multi-GPUs with single precision.
        Using when the c_index is higher than the maximum qubits in the current
        device, and the t_index doesn't.
        """
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_t(
            (block_num,),
            (thread_per_block,),
            (t_index, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()
