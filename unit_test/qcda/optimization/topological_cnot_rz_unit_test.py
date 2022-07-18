#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/30 2:54
# @Author  : Han Yu
# @File    : unit_test.py

import random

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.core.gate import CX, Rz
from QuICT.core.layout import Layout
from QuICT.qcda.optimization import TopologicalCnotRz


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


def check_equiv(circuit1, circuit2):
    """ check whether two circuit is equiv

    Args:
        circuit1(Circuit)
        circuit2(Circuit/list<BasicGate>)
    Returns:
        bool: True if equiv
    """
    matrix1 = SyntheticalUnitary.run(circuit1)
    matrix2 = SyntheticalUnitary.run(circuit2)
    return not np.any(abs(abs(matrix1 - matrix2)) > 1e-6)


def test_1():
    for _ in range(1):
        for i in range(2, 6):
            layout = Layout(i)
            topo = _getAllRandomList(i)
            for j in range(len(topo) - 1):
                layout.add_edge(topo[j], topo[j + 1])
            for _ in range(i // 10):
                layout.add_edge(_getRandomList(2))
            circuit = Circuit(i, topology=layout)
            for j in range(i * 8):
                CX | circuit(list(_getRandomList(2)))
                if j % 10 == 0:
                    Rz(random.random() * np.pi) | circuit(random.randrange(0, i))
            new_circuit = TopologicalCnotRz.execute(circuit)
            if not check_equiv(circuit, new_circuit):
                assert 0


def test_2():
    circuit = Circuit(2)
    CX | circuit([0, 1])
    Rz(np.pi / 4) | circuit(1)
    new_circuit = TopologicalCnotRz.execute(circuit)
    if not check_equiv(circuit, new_circuit):
        assert 0
