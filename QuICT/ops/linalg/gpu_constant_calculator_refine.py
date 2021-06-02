#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 下午3:34
# @Author  : Han Yu
# @File    : gpu_constant_calculator_refine

import numba
from numba import cuda
import numpy as np
import cmath

from QuICT.core import *

@cuda.jit()
def _H_large_vec_kernel(
    index: int,
    vec
):
    label = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    _0 = (label & ((1 << index) - 1)) + (label >> index << (index + 1))
    _1 = _0 + (1 << index)
    vec[_0], vec[_1] = (vec[_0] + vec[_1]) / cmath.sqrt(2), (vec[_0] - vec[_1]) / cmath.sqrt(2)

@cuda.jit()
def _CRz_large_vec_kernel1(
    cindex,
    tindex,
    vec,
    ww
):
    label = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    gw = label >> cindex << (cindex + 1)
    _0 = (1 << cindex) + (gw & ((1 << tindex) - (1 << cindex))) + (gw >> tindex << (tindex + 1)) + \
         (label & ((1 << cindex) - 1))
    _1 = _0 + (1 << tindex)
    vec[_0] = vec[_0] * ww
    vec[_1] = vec[_1] * ww.conjugate()

@cuda.jit()
def _CRz_large_vec_kernel2(
    cindex,
    tindex,
    vec,
    ww
):
    label = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    gw = label >> tindex << (tindex + 1)
    _0 = (1 << cindex) + (gw & ((1 << cindex) - (1 << tindex))) + (gw >> cindex << (cindex + 1)) + \
         (label & ((1 << tindex) - 1))
    _1 = _0 + (1 << tindex)
    vec[_0] = vec[_0] * ww
    vec[_1] = vec[_1] * ww.conjugate()

def gate_dot_vector_cuda(
    gate : BasicGate,
    vec,
    vec_bit
):
    if gate.type() == GATE_ID["H"]:

        task_number = 1 << (vec_bit - 1)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        _H_large_vec_kernel[[block_num, 1, 1], [thread_per_block, 1, 1]](
            vec_bit - 1 - gate.targ,
            vec
        )
    elif gate.type() == GATE_ID["CRz"]:
        cindex = vec_bit - 1 - gate.carg
        tindex = vec_bit - 1 - gate.targ

        task_number = 1 << (vec_bit - 2)
        thread_per_block = min(256, task_number)
        block_num = task_number // thread_per_block

        if tindex > cindex:
            _CRz_large_vec_kernel1[[block_num, 1, 1], [thread_per_block, 1, 1]](
                cindex,
                tindex,
                vec,
                np.exp(-0.5j * gate.parg)
            )
        else:
            _CRz_large_vec_kernel2[[block_num, 1, 1], [thread_per_block, 1, 1]](
                cindex,
                tindex,
                vec,
                np.exp(-0.5j * gate.parg)
            )
    else:
        raise Exception("ss")
