#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 11:50 下午
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random
import numpy as np

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis import UniformlyRy, UniformlyRz, UniformlyUnitary


def generate_unitary():
    matrix = U3(random.random() * np.pi, random.random() * np.pi, random.random() * np.pi).matrix
    matrix[:] *= np.exp(2j * np.pi * random.random())

    return matrix


def test_uniform_ry():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            angles = [random.random() for _ in range(1 << (i - 1))]
            UniformlyRy.execute(angles) | circuit
            unitary = circuit.matrix()
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert not np.any(abs(unitary_slice - Ry(angles[j]).matrix.reshape(2, 2)) > 1e-10)


def test_uniform_rz():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            angles = [random.random() for _ in range(1 << (i - 1))]
            UniformlyRz.execute(angles) | circuit
            unitary = circuit.matrix()
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert not np.any(abs(unitary_slice - Rz(angles[j]).matrix.reshape(2, 2)) > 1e-10)


def test_uniform_unitary():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            unitaries = [generate_unitary() for _ in range(1 << (i - 1))]
            UniformlyUnitary.execute(unitaries) | circuit
            unitary = circuit.matrix()
            if abs(unitary[0, 0]) > 1e-10:
                delta = unitaries[0][0][0] / unitary[0, 0]
            else:
                delta = unitaries[0][0][1] / unitary[0, 1]
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                unitary_slice[:] *= delta
                phase = np.any(abs(unitary_slice - unitaries[j].reshape(2, 2)) > 1e-6)
                if phase:
                    assert 0


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
