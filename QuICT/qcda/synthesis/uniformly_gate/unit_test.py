#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 11:50 下午
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.synthesis import uniformlyRy, uniformlyRz

def test_uniform_ry():
    for i in range(1, 8):
        circuit = Circuit(i)
        angles = [random.random() for _ in range(1 << (i - 1))]
        uniformlyRy(angles) | circuit
        unitary = SyntheticalUnitary.run(circuit)
        for j in range(1 << (i - 1)):
            unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
            assert not np.any(abs(unitary_slice - Ry(angles[j]).matrix.reshape(2, 2)) > 1e-10)

def test_uniform_rz():
    for i in range(1, 8):
        circuit = Circuit(i)
        angles = [random.random() for _ in range(1 << (i - 1))]
        uniformlyRz(angles) | circuit
        unitary = SyntheticalUnitary.run(circuit)
        for j in range(1 << (i - 1)):
            unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
            assert not np.any(abs(unitary_slice - Rz(angles[j]).matrix.reshape(2, 2)) > 1e-10)

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
