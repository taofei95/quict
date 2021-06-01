#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/27 8:36 下午
# @Author  : Han Yu
# @File    : gpu_calculator_refine.py

import os
import math
import numpy as np

from typing import *

from numba import cuda as nb_cuda

from time import time

def htod(target):
    """ mv target from host into GPU device. """
    return nb_cuda.to_device(target)

@nb_cuda.jit()
def _small_mat_large_vec_kernel(
    mat,
    matrix_bit: int,
    matrix_length: int,
    vec,
    affect_args,
    affect_args_sorted,
    auxiliary_vec
):
    bx = nb_cuda.blockIdx.x
    tx = nb_cuda.threadIdx.x
    bw = nb_cuda.blockDim.x
    label = bx * bw + tx
    other = label & ((1 << affect_args_sorted[0]) - 1)
    gw = label >> affect_args_sorted[0] << (affect_args_sorted[0] + 1)
    for i in range(1, matrix_bit):
       other += gw & ((1 << affect_args_sorted[i]) - (1 << affect_args_sorted[i - 1]))
       gw = gw >> affect_args_sorted[i] << (affect_args_sorted[i] + 1)
    other += gw
    for i in range(matrix_length):
        now = other
        for k in range(matrix_bit):
            if i & (1 << k):
                now += 1 << affect_args[matrix_bit - 1 - k]
        for k in range(matrix_length):
            shift = other
            for l in range(matrix_bit):
                if k & (1 << l):
                    shift += 1 << affect_args[matrix_bit - 1 - l]
            auxiliary_vec[now] += mat[i, k] * vec[shift]

def matrix_dot_vector_cuda(
    mat,
    mat_bit,
    vec,
    vec_bit,
    affect_args
):
    vec_length = 1 << vec_bit
    mat_length = 1 << mat_bit

    task_number = 1 << (vec_bit - mat_bit)
    thread_per_block = min(256, task_number)
    block_num = task_number // thread_per_block

    for i in range(mat_bit):
       affect_args[i] = vec_bit - 1 - affect_args[i]

    affect_args_sorts = affect_args.copy()
    affect_args_sorts.sort()

    htod_time_start = time()

    mat = htod(mat)
    mat_bit = np.int32(mat_bit)
    mat_length = np.int64(mat_length)
    affect_args = htod(affect_args)
    affect_args_sorts = htod(affect_args_sorts)

    htod_time_end = time()

    auxiliary_vec = htod(np.zeros([vec_length, ], dtype=np.complex64))

    cal_time_start = time()

    _small_mat_large_vec_kernel[[block_num, 1, 1], [thread_per_block, 1, 1]](
        mat,
        mat_bit,
        mat_length,
        vec,
        affect_args,
        affect_args_sorts,
        auxiliary_vec
    )

    cal_time_end = time()
    print(f"{mat_bit}-qubit total", htod_time_end - htod_time_start + cal_time_end - cal_time_start)
    print("matrix_htod", htod_time_end - htod_time_start)
    print("cal", cal_time_end - cal_time_start)

    return auxiliary_vec

