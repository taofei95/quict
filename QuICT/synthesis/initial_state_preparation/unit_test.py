#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51 上午
# @Author  : Han Yu
# @File    : _unit_test.py

from QuICT.models import *
from QuICT.algorithm import *
from QuICT.synthesis import initial_state_preparation
import numpy as np
import pytest

def w_test_1():
    for i in range(8):
        print(i)
        circuit = Circuit(i)
        values = [1.0 / (1 << i) for _ in range(1 << i)]
        initial_state_preparation(values) | circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        for k in range(1 << i):
            if abs(abs(amplitude[k]) * abs(amplitude[k]) - (1.0 / (1 << i))) > 1e-6:
                print(i, k, abs(amplitude[k]) * abs(amplitude[k]), (1.0 / (1 << i)))
                assert 0
    assert 1

def w_test_2():
    for i in range(8):
        print(i)
        circuit = Circuit(i)
        values = [1.0 / (1 << i) for _ in range(1 << i)]
        initial_state_preparation(values) | circuit([j for j in range(i)])
        initial_state_preparation(values) ^ circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        if abs(amplitude[0] - 1) > 1e-10:
            assert 0
    assert 1

def test_3():
    circuit = Circuit(2)
    qreg = circuit([0, 1])
    p = np.array([0, 0, 0, 1])
    print(p)
    initial_state_preparation(list(p)) | qreg
    circuit.print_infomation()
    amp = Amplitude.run(circuit)
    print(amp)
    assert 0


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
