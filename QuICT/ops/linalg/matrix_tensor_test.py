#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/20 11:35 下午
# @Author  : Han Yu
# @File    : unit_test.py

import time

from numba import njit, prange
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pyopencl as cl
import pyopencl.tools
import pyopencl.array

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
def dot_np_numba(A, B):
    return np.dot(A, B)

def dot_pycuda(A, B):
    a = gpuarray.to_gpu(A)
    b = gpuarray.to_gpu(B)
    return gpuarray.dot(a, b).get()

def dot_pyopencl(A, B):
    # ctx = cl.create_some_context()
    platforms = cl.get_platforms()
    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])
    queue = cl.CommandQueue(ctx)
    a = cl.array.to_device(queue, A)
    b = cl.array.to_device(queue, B)
    return cl.array.dot(a, b).get()

for i in range(10, 14):
    f_m = open(f"matrix_qubit_{i}.txt", "w")
    f_tI = open(f"tensor_qubit_{i}.txt", "w")
    f_tM = open(f"tensor_qubit_{i}.txt", "w")
    n = 1 << i
    A = np.random.random([n, n])
    B = np.random.random([n, n])
    log_time(f_m, np.dot, A, B, "dot")
    log_time(f_m, dot_np_numba, A, B, "dot with numba")
    log_time(f_m, dot_pycuda, A, B, "dot with pycuda")
    log_time(f_m, dot_pyopencl, A, B, "dot with pyopencl")
