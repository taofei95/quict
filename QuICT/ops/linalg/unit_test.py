#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/20 11:35 下午
# @Author  : Han Yu
# @File    : unit_test.py

import time

from numba import njit, prange
import numpy as np
from scipy.stats import unitary_group
from random import shuffle

from .unitary_calculation import *


def log_time(f, func, A, B, name=None):
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


# for i in range(7, 16):
#     f_m = open(f"matrix_qubit_{i}.txt", "w")
#     f_tI = open(f"tensor_qubit_{i}.txt", "w")
#     f_tM = open(f"tensor_qubit_{i}.txt", "w")
#     n = 1 << i
#     A = np.random.random([n, n])
#     B = np.random.random([n, n])
#     log_time(f_m, np.dot, A, B, "dot")
#     log_time(f_m, dot_np_numba, A, B, "dot with numba")

def get_random_complex_matrix(n):
    return np.random.rand(n, n) + np.random.rand(n, n) * 1j


def test_multiplication_performance():
    qubit_num = 10
    print()
    print("First run for JIT...")
    a = get_random_complex_matrix(1 << qubit_num)
    b = get_random_complex_matrix(1 << qubit_num)
    _ = dot(a, b)

    print("Tests start here...")

    for qubit_num in range(10, 15):
        rnd = 5
        total_time = 0.0
        for cur_rnd in range(rnd):
            a = get_random_complex_matrix(1 << qubit_num)
            b = get_random_complex_matrix(1 << qubit_num)
            start_time = time.time()
            _ = dot(a, b)
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # in ms
        print(f"qubit_num = {qubit_num}, average multiplication time = {total_time / rnd: 0.4f} ms, in {rnd} round(s)")


def test_permute_performance():
    qubit_num = 10
    print()
    print("First run for JIT...")
    mat = get_random_complex_matrix(1 << qubit_num)
    perm = [i for i in range(qubit_num)]
    shuffle(perm)
    perm = np.array(perm)
    _ = MatrixPermutation(mat, perm)

    print("Tests start here...")

    for qubit_num in range(10, 15):
        rnd = 5
        total_time = 0.0
        for cur_rnd in range(rnd):
            mat = get_random_complex_matrix(1 << qubit_num)
            perm = [i for i in range(qubit_num)]
            shuffle(perm)
            perm = np.array(perm)
            start_time = time.time()
            _ = MatrixPermutation(mat, perm)
            end_time = time.time()
            total_time += (end_time - start_time) * 1000  # in ms
        print(f"qubit_num = {qubit_num}, average permutation time = {total_time / rnd: 0.4f} ms, in {rnd} round(s)")
