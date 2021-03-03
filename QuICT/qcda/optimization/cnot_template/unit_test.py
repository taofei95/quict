#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/22 9:56
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.optimization import CnotForceBfs, CnotForceDepthBfs, CnotLocalForceBfs, CnotStoreForceBfs,\
    CnotLocalForceDepthBfs


def _getRandomList(n):
    """ get first 2 number from 0, 1, ..., n - 1 randomly.
    Args:
        n(int)
    Returns:
        tuple<int, int>
    """
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[0], _rand[1]

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
    n = circuit1.circuit_width()
    if circuit2.circuit_width() != n:
        return False
    matrix1 = generate_matrix(circuit1, n)
    matrix2 = generate_matrix(circuit2, n)

    return not np.any(matrix1 ^ matrix2)

def w_test_1():
    for i in range(5, 6):
        circuit = Circuit(i)
        for _ in range(10000):
            cx = _getRandomList(i)
            CX | circuit(cx)
        new_circuit = CnotForceBfs.run(circuit)
        if not check_equiv(circuit, new_circuit):
            assert 0

def w_test_2():
    for i in range(5, 6):
        circuit = Circuit(i)
        for _ in range(1000):
            cx = _getRandomList(i)
            Z  | circuit(random.randrange(i))
            S  | circuit(random.randrange(i))
            T  | circuit(random.randrange(i))
            X  | circuit(random.randrange(i))
            CX | circuit(cx)
        new_circuit = CnotLocalForceBfs.run(circuit, True)
        circuit.print_infomation()
        new_circuit.print_infomation()
        syn1 = SyntheticalUnitary.run(circuit)
        syn2 = SyntheticalUnitary.run(new_circuit)
        assert not np.any(np.abs(syn1 - syn2) > 1e-7)

def w_test_3():
    for _ in range(1):
        for i in range(6, 7):
            circuit = Circuit(i)
            for _ in range(100):
                cx = _getRandomList(i)
                CX | circuit(cx)
            new_circuit = CnotLocalForceBfs.run(circuit, True)
            if not check_equiv(circuit, new_circuit):
                assert 0

def w_test_4():
    for i in range(2, 4):
        circuit = Circuit(i)
        for _ in range(10000):
            cx = _getRandomList(i)
            CX | circuit(cx)
        new_circuit = CnotForceDepthBfs.run(circuit)
        if not check_equiv(circuit, new_circuit):
            assert 0

def test_5():
    for i in range(5, 6):
        circuit = Circuit(i)
        for _ in range(10000):
            cx = _getRandomList(i)
            CX | circuit(cx)
        new_circuit = CnotLocalForceDepthBfs.run(circuit, True)
        if not check_equiv(circuit, new_circuit):
            assert 0

if __name__ == '__main__':
    # pytest.main(["./unit_test.py"])
    test_5()
