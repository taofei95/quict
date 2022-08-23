#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/30 2:54
# @Author  : Han Yu
# @File    : unit_test.py

import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CX, Rz
from QuICT.core.layout import Layout
from QuICT.qcda.optimization import TopologicalCnotRz


def check_equiv(circuit1, circuit2):
    """ check whether two circuit is equiv

    Args:
        circuit1(Circuit)
        circuit2(Circuit/list<BasicGate>)
    Returns:
        bool: True if equiv
    """
    matrix1 = circuit1.matrix()
    matrix2 = circuit2.matrix()
    return np.allclose(matrix1, matrix2)


def test():
    for _ in range(20):
        for n in range(2, 6):
            layout = Layout(n)
            topo = random.sample(range(n), n)
            for j in range(len(topo) - 1):
                layout.add_edge(topo[j], topo[j + 1])
            for _ in range(n // 10):
                layout.add_edge(random.sample(range(n), 2))
            circuit = Circuit(n, topology=layout)
            for j in range(n * 8):
                CX | circuit(random.sample(range(n), 2))
                if j % 10 == 0:
                    Rz(random.random() * np.pi) | circuit(random.randrange(0, n))
            TCR = TopologicalCnotRz()
            new_circuit = TCR.execute(circuit)
            assert check_equiv(circuit, new_circuit)
