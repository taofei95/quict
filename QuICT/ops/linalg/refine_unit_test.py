#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/27 8:36 下午
# @Author  : Han Yu
# @File    : refine_unit_test

from QuICT.core import *
from QuICT.algorithm import Amplitude

from scipy.stats import ortho_group

import numpy as np
import numba

import pytest

import time
from QuICT.ops.linalg.gpu_calculator_refine import matrix_dot_vector_cuda

def test_refine():
    vec_bit = 20
    vec = np.random.random((1 << vec_bit,)).astype(np.complex64)
    anc = np.zeros((1 << vec_bit, )).astype(np.complex64)
    vec_htod_start = time.time()
    vec = numba.cuda.to_device(vec)
    vec_htod_end = time.time()
    # print("vec_htod_time", vec_htod_end - vec_htod_start)
    for i in range(10, 12):
        mat_bit = i
        mat = np.random.random((1 << mat_bit, 1 << mat_bit)).astype(np.complex64)

        affect_args = np.array([i for i in range(vec_bit)], dtype=np.int32)
        np.random.shuffle(affect_args)
        affect_args = affect_args[:mat_bit]

        with numba.cuda.gpus[0]:
            matrix_dot_vector_cuda(
                mat = mat,
                mat_bit = mat_bit,
                vec = vec,
                vec_bit = vec_bit,
                affect_args = affect_args,
                auxiliary_vec = anc
            )
            vec, anc = anc, vec
            dtot_time_start = time.time()
            # ans.copy_to_host(ans_arr)
            dtot_time_end = time.time()
            # print("dtoh", dtot_time_end - dtot_time_start)
            numba.cuda.synchronize()
    ans_arr = np.zeros_like(vec, dtype=np.complex64)
    dtot_time_start = time.time()
    with numba.cuda.gpus[0]:
        vec.copy_to_host(ans_arr)
    dtot_time_end = time.time()
    # print("vec dtoh", dtot_time_end - dtot_time_start)
    assert 0
