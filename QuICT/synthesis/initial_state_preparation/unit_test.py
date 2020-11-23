#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51 上午
# @Author  : Han Yu
# @File    : _unit_test.py

import pytest
import random

import numpy as np

from QuICT.models import *
from QuICT.algorithm import *
from QuICT.synthesis import initial_state_preparation

def test_1():
    for i in range(8):
        circuit = Circuit(i)
        values = [1.0 / (1 << i) for _ in range(1 << i)]
        initial_state_preparation(values) | circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        for k in range(1 << i):
            if abs(abs(amplitude[k]) * abs(amplitude[k]) - (1.0 / (1 << i))) > 1e-6:
                print(i, k, abs(amplitude[k]) * abs(amplitude[k]), (1.0 / (1 << i)))
                assert 0
    assert 1

def test_2():
    for i in range(8):
        circuit = Circuit(i)
        values = [1.0 / (1 << i) for _ in range(1 << i)]
        initial_state_preparation(values) | circuit([j for j in range(i)])
        initial_state_preparation(values) ^ circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        if abs(amplitude[0] - 1) > 1e-10:
            assert 0
    assert 1

def w_test_3():
    circuit = Circuit(3)
    values = [0.25, 0.1, 0.25, 0.25, 0, 0.25, 0]
    # Ry(np.pi) | circuit(0)
    initial_state_preparation(values) | circuit
    # Ry(np.pi / 2) | circuit(2)
    # Ry(np.pi / 2) | circuit(2)
    amplitude = Amplitude.run(circuit)
    circuit.print_infomation()
    print(amplitude)
    assert 0

def test_4():
    for i in range(8):
        circuit = Circuit(i)
        values = [random.random() for i in range(1 << i)]
        initial_state_preparation(values) | circuit([j for j in range(i)])
        initial_state_preparation(values) ^ circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        if abs(amplitude[0] - 1) > 1e-10:
            assert 0
    assert 1

def test_5():
    for i in range(8):
        circuit = Circuit(i)
        values = [random.random() for i in range(1 << i)]
        norm = 0
        for v in values:
            norm += v
        initial_state_preparation(values) | circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        for j in range(1 << i):
            values[j] /= norm
        for j in range(1 << i):
            if abs(abs(np.sqrt(values[j])  - amplitude[j])) > 1e-6:
                assert 0
    assert 1

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
