#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/22 9:56
# @Author  : Han Yu
# @File    : unit_test.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import GateType
from QuICT.qcda.optimization import CnotForceBfs, CnotForceDepthBfs, CnotLocalForceBfs, CnotLocalForceDepthBfs


def generate_matrix(circuit, n):
    matrix = np.identity(n, dtype=bool)
    for gate in circuit.gates:
        matrix[gate.targ, :] = matrix[gate.targ, :] ^ matrix[gate.carg, :]
    return matrix


def check_equiv(circuit1, circuit2):
    """ check whether two circuit is equiv

    Args:
        circuit1(Circuit)
        circuit2(Circuit)
    Returns:
        bool: True if equiv
    """
    n = circuit1.width()
    if circuit2.width() != n:
        return False
    matrix1 = generate_matrix(circuit1, n)
    matrix2 = generate_matrix(circuit2, n)

    return not np.any(matrix1 ^ matrix2)


def test_1():
    for i in range(2, 4):
        circuit = Circuit(i)
        circuit.random_append(10, typelist=[GateType.cx])
        CFB = CnotForceBfs()
        new_circuit = CFB.execute(circuit)
        assert check_equiv(circuit, new_circuit)


def test_2():
    for i in range(5, 6):
        circuit = Circuit(i)
        circuit.random_append(30, typelist=[GateType.z, GateType.x, GateType.cx])
        CLFB = CnotLocalForceBfs(False)
        new_circuit = CLFB.execute(circuit)
        syn1 = circuit.matrix()
        syn2 = new_circuit.matrix()
        assert np.allclose(syn1, syn2)


def test_3():
    for i in range(4, 7):
        circuit = Circuit(i)
        circuit.random_append(10, typelist=[GateType.cx])
        CLFB = CnotLocalForceBfs(False)
        new_circuit = CLFB.execute(circuit)
        assert check_equiv(circuit, new_circuit)


def test_4():
    for i in range(2, 4):
        circuit = Circuit(i)
        circuit.random_append(10, typelist=[GateType.cx])
        CFDB = CnotForceDepthBfs()
        new_circuit = CFDB.execute(circuit)
        assert check_equiv(circuit, new_circuit)


def test_5():
    for i in range(4, 7):
        circuit = Circuit(i)
        circuit.random_append(10, typelist=[GateType.cx])
        CLFDB = CnotLocalForceDepthBfs(False)
        new_circuit = CLFDB.execute(circuit)
        assert check_equiv(circuit, new_circuit)
