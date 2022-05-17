#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 11:50 下午
# @Author  : Han Yu
# @File    : unit_test.py

import random

import numpy as np
from scipy.stats import unitary_group

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis import UniformlyRotation, UniformlyUnitary


def test_uniformly_ry():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            angles = [random.random() for _ in range(1 << (i - 1))]
            URy = UniformlyRotation(GateType.ry)
            URy.execute(angles) | circuit
            unitary = SyntheticalUnitary.run(circuit)
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert np.allclose(unitary_slice, Ry(angles[j]).matrix)


def test_uniformly_rz():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            angles = [random.random() for _ in range(1 << (i - 1))]
            URz = UniformlyRotation(GateType.rz)
            URz.execute(angles) | circuit
            unitary = SyntheticalUnitary.run(circuit)
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert np.allclose(unitary_slice, Rz(angles[j]).matrix)


def test_uniformly_unitary():
    for _ in range(10):
        for i in range(1, 6):
            circuit = Circuit(i)
            unitaries = [unitary_group.rvs(2) for _ in range(1 << (i - 1))]
            UUnitary = UniformlyUnitary()
            UUnitary.execute(unitaries) | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if abs(unitary[0, 0]) > 1e-10:
                delta = unitaries[0][0][0] / unitary[0, 0]
            else:
                delta = unitaries[0][0][1] / unitary[0, 1]
            for j in range(1 << (i - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                unitary_slice[:] *= delta
                assert np.allclose(unitary_slice, unitaries[j].reshape(2, 2))
