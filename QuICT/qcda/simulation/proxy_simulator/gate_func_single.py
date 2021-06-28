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
    @classmethod
    def HGate_matrixdot(self, targ, mat, vec, vec_bit, sync: bool = True):
        print(mat)
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block
        HGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (vec_bit - 1 - targ, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot(self, carg, targ, mat, vec, vec_bit, sync: bool = True):
        cindex = vec_bit - 1 - carg
        tindex = vec_bit - 1 - targ

        task_number = 1 << (vec_bit - 2)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_single_kernel(
            (block_num,),
            (thread_per_block,),
            (cindex, tindex, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()


class GateFuncMS:
    @classmethod
    def CRzGate_matrixdot_pd(self, _0_1, mat, vec, vec_bit, sync: bool = True):
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
    def CRzGate_matrixdot_pc(self, _0_1, carg, mat, vec, vec_bit, sync: bool = True):
        mat_value = mat[10] if _0_1 else mat[15]
        cindex = vec_bit - 1 - carg

        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_c(
            (block_num,),
            (thread_per_block,),
            (cindex, mat_value, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()

    @classmethod
    def CRzGate_matrixdot_pt(self, tindex, mat, vec, vec_bit, sync: bool = True):
        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        CRZGate_kernel_special_t(
            (block_num,),
            (thread_per_block,),
            (tindex, mat, vec)
        )

        if sync:
            cp.cuda.Device().synchronize()
