#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/26 12:25
# @Author  : Han Yu
# @File    : unit_test.py

import numpy as np
import pytest
import random

from QuICT.core import *
from QuICT.qcda.optimization import CnotAncillae

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

def generate_matrix_with_ancillary(circuit, n):
    circuit_length = circuit.circuit_width()
    matrix = np.identity(circuit_length, dtype=bool)
    for gate in circuit.gates:
        matrix[gate.targ, :] = matrix[gate.targ, :] ^ matrix[gate.carg, :]
    for i in range(n):
        for j in range(n):
            assert not matrix[i, j]
    for i in range(2 * n, 3 * n):
        for j in range(n):
            assert not matrix[i, j]
    return matrix[n:2 * n, :n]

def check_equiv(circuit1, circuit2):
    """ check whether two circuit is equiv

    Args:
        circuit1(Circuit)
        circuit2(Circuit)
    Returns:
        bool: True if equiv
    """
    n = circuit1.circuit_width()
    matrix1 = generate_matrix(circuit1, n)
    matrix2 = generate_matrix_with_ancillary(circuit2, n)
    # circuit2.print_information()

    print(matrix1)

    print(np.any(matrix1 ^ matrix2))

    return not np.any(matrix1 ^ matrix2)

def test_1():
    for n in range(4, 100):
        for s in range(1, int(np.floor(n / np.log2(n) / np.log2(n)))):
            circuit = Circuit(n)
            for i in range(n - 1):
                CX | circuit([i, i + 1])
            new_circuit = CnotAncillae.run(circuit, size = s)
            assert check_equiv(circuit, new_circuit)

def test_2():
    assert 1

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
