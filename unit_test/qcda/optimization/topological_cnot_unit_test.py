#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/30 2:54
# @Author  : Han Yu
# @File    : unit_test.py

import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CX, GateType
from QuICT.core.layout import Layout
from QuICT.qcda.optimization import TopologicalCnot


def generate_matrix(gates, n):
    matrix = np.identity(n, dtype=bool)
    i = 0
    lg = len(gates)
    while i < lg:
        gate = gates[i]
        if gate.type == GateType.h:
            gate = gates[i + 2]
            i += 4
            matrix[gate.carg, :] = matrix[gate.carg, :] ^ matrix[gate.targ, :]
        else:
            matrix[gate.targ, :] = matrix[gate.targ, :] ^ matrix[gate.carg, :]
        i += 1
    return matrix


def check_equiv(circuit1, circuit2):
    """ check whether two circuit is equiv

    Args:
        circuit1(Circuit)
        circuit2(Circuit/list<BasicGate>)
    Returns:
        bool: True if equiv
    """
    n = circuit1.width()
    matrix1 = generate_matrix(circuit1.gates, n)
    matrix2 = generate_matrix(circuit2.gates, n)
    return not np.any(matrix1 ^ matrix2)


def test():
    for _ in range(20):
        for n in range(2, 10):
            layout = Layout(n)
            topo = random.sample(range(n), n)
            for j in range(len(topo) - 1):
                layout.add_edge(topo[j], topo[j + 1])
            for _ in range(n // 10):
                layout.add_edge(random.sample(range(n), 2))
            circuit = Circuit(n, topology=layout)
            for _ in range(n * 100):
                CX | circuit(list(random.sample(range(n), 2)))
            TC = TopologicalCnot()
            new_circuit = TC.execute(circuit)
            assert check_equiv(circuit, new_circuit)
