#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51 上午
# @Author  : Han Yu
# @File    : _unit_test.py

from QuICT.core import *
from QuICT.algorithm import *
from QuICT.qcda.synthesis import InitialStatePreparation
import numpy as np
import pytest

def test_1():
    for i in range(8):
        print(i)
        circuit = Circuit(i)
        values = [1.0 / (1 << i) for _ in range(1 << i)]
        InitialStatePreparation(values) | circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        for k in range(1 << i):
            if abs(abs(amplitude[k]) * abs(amplitude[k]) - (1.0 / (1 << i))) > 1e-6:
                print(i, k, abs(amplitude[k]) * abs(amplitude[k]), (1.0 / (1 << i)))
                assert 0
    assert 1

def test_2():
    for i in range(8):
        print(i)
        circuit = Circuit(i)
        values = [1.0 / (1 << i) for _ in range(1 << i)]
        InitialStatePreparation(values) | circuit([j for j in range(i)])
        InitialStatePreparation(values) ^ circuit([j for j in range(i)])
        amplitude = Amplitude.run(circuit)
        if abs(amplitude[0] - 1) > 1e-10:
            assert 0
    assert 1


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
