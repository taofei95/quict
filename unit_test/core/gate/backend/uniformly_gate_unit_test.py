#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 11:50 下午
# @Author  : Han Yu
# @File    : unit_test.py

import numpy as np
from scipy.stats import unitary_group

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.core.gate.backend import UniformlyRotation, UniformlyUnitary


def test_uniformly_ry():
    for _ in range(10):
        for n in range(1, 6):
            circuit = Circuit(n)
            angles = [2 * np.pi * np.random.random() for _ in range(1 << (n - 1))]
            URy = UniformlyRotation(GateType.ry)
            URy.execute(angles) | circuit
            unitary = circuit.matrix()
            for j in range(1 << (n - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert np.allclose(unitary_slice, Ry(angles[j]).matrix)


def test_uniformly_rz():
    for _ in range(10):
        for n in range(1, 6):
            circuit = Circuit(n)
            angles = [2 * np.pi * np.random.random() for _ in range(1 << (n - 1))]
            URz = UniformlyRotation(GateType.rz)
            URz.execute(angles) | circuit
            unitary = circuit.matrix()
            for j in range(1 << (n - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                assert np.allclose(unitary_slice, Rz(angles[j]).matrix)


def test_uniformly_unitary():
    for _ in range(10):
        for n in range(1, 6):
            circuit = Circuit(n)
            unitaries = [unitary_group.rvs(2) for _ in range(1 << (n - 1))]
            UUnitary = UniformlyUnitary()
            UUnitary.execute(unitaries) | circuit
            unitary = circuit.matrix()
            if abs(unitary[0, 0]) > 1e-10:
                delta = unitaries[0][0][0] / unitary[0, 0]
            else:
                delta = unitaries[0][0][1] / unitary[0, 1]
            for j in range(1 << (n - 1)):
                unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
                unitary_slice[:] *= delta
                assert np.allclose(unitary_slice, unitaries[j].reshape(2, 2))
