#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/20 11:35 下午
# @Author  : Han Yu
# @File    : unit_test.py

import time

from numba import njit, prange
import numpy as np

from unitary_calculation import *

def log_time(f, func, A, B, name = None):
    if name is None:
        name = func.__name__
    t1 = time.time()
    C = func(A, B)
    t2 = time.time()
    f.write(f"{name} time pre jit:{(t2 - t1)}\n")
    t1 = time.time()
    C = func(A, B)
    t2 = time.time()
    f.write(f"{name} time after jit:{t2 - t1}\n")

@njit()
def kron_np_numba(A, B):
    return np.kron(A, B)

for i in range(7, 16):
    f_tI = open(f"tensor_qubit_{i}.txt", "w")
    i1 = i // 2
    i2 = i - i1
    n = 1 << i
    A = np.random.random([1 << i1, 1 << i1])
    B = np.random.random([1 << i2, 1 << i2])
    if i <= 11:
        log_time(f_tI, tensor, A, B, "tensor")
    log_time(f_tI, np.kron, A, B, "tensor")
    log_time(f_tI, kron_np_numba, A, B, "tensor with numba")
