#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/30 2:54
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random

import numpy as np

from QuICT.core import *
from QuICT.qcda.optimization import TopologicalCnot


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


def _getAllRandomList(n):
    """ get n number from 0, 1, ..., n - 1 randomly.

    Args:
        n(int)
    Returns:
        tuple<int, int>
    """
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand


def generate_matrix(gates, n):
    matrix = np.identity(n, dtype=bool)
    i = 0
    lg = len(gates)
    while i < lg:
        gate = gates[i]
        if gate.type() == GATE_ID["H"]:
            gate = gates[i + 2]
            i += 4
            matrix[gate.carg, :] = matrix[gate.carg, :] ^ matrix[gate.targ, :]
        else:
            matrix[gate.targ, :] = matrix[gate.targ, :] ^ matrix[gate.carg, :]
        i += 1
    return matrix


def generate_matrix_list(gates, n):
    matrix = generate_matrix(gates, n)
    matrix_values = []
    for i in range(n):
        values = 0
        for j in range(n):
            if matrix[i, j]:
                values += 1 << j
        matrix_values.append(values)
    return matrix_values


def check_equiv(circuit1, circuit2):
    """ check whether two circuit is equiv

    Args:
        circuit1(Circuit)
        circuit2(Circuit/list<BasicGate>)
    Returns:
        bool: True if equiv
    """
    n = circuit1.circuit_width()
    matrix1 = generate_matrix(circuit1.gates, n)
    matrix2 = generate_matrix(circuit2.gates if isinstance(circuit2, Circuit) else circuit2, n)

    return not np.any(matrix1 ^ matrix2)


def test_1():
    for _ in range(20):
        for i in range(2, 10):
            circuit = Circuit(i)
            for _ in range(i * 100):
                CX | circuit(_getRandomList(2))
            topo = _getAllRandomList(i)
            for j in range(len(topo) - 1):
                circuit.add_topology((topo[j], topo[j + 1]))
            for _ in range(i // 10):
                circuit.add_topology(_getRandomList(2))
            new_circuit = TopologicalCnot.execute(circuit)
            if not check_equiv(circuit, new_circuit):
                assert 0

            circuit = Circuit(i)
            for _ in range(i * 100):
                CX | circuit(_getRandomList(2))
            topo = _getAllRandomList(i)
            topology = []
            for j in range(len(topo) - 1):
                topology.append((topo[j], topo[j + 1]))
            for _ in range(i // 10):
                topology.append(_getRandomList(2))
            new_circuit = TopologicalCnot.execute(cnot_struct=generate_matrix_list(circuit.gates, i), topology=topology)
            if not check_equiv(circuit, new_circuit):
                assert 0


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
