#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51
# @Author  : Han Yu
# @File    : _unit_test.py

import pytest
import random

import numpy as np

from QuICT.core import *
from QuICT.algorithm import *
from QuICT.qcda.synthesis import InitialStatePreparation

def check_assert(a, b, n):
    norm = 0
    for value in b:
        norm += abs(value) * abs(value)
    for i in range(n):
        b[i] /= norm
    flag = None
    for i in range(n):
        if abs(a[i] - b[i]) >= 1e-6:
            flag = a[i] / b[i]
            break
    if flag is None:
        return True
    for i in range(n):
        if abs(a[i] - b[i] * flag) >= 1e-6:
            return False
    return True

def test_1():
    for i in range(1, 8):
        circuit = Circuit(i)
        if i != 3:
            values = [1.0 / (1 << i) * np.exp(1j * random.random()) for _ in range(1 << i)]
        else:
            values = [0.40824829046386296, 0.40824829046386296, 0.40824829046386296, 0.40824829046386296,
                  0.40824829046386296, 0, 0.2886751345948131, -0.2886751345948128]
        InitialStatePreparation(values) | circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        now = check_assert(amplitude, values, 1 << i)
        print(amplitude)
        print(i)
        assert now

def w_test_2():
    for i in range(1, 8):
        circuit = Circuit(i)
        values = [1.0 / (1 << i) * np.exp(1j * random.random()) for _ in range(1 << i)]
        InitialStatePreparation(values) | circuit([j for j in range(i)])
        InitialStatePreparation(values) ^ circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        if abs(amplitude[0] - 1) > 1e-6:
            assert 0

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
    # let_test()
