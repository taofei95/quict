#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 10:43
# @Author  : Han Yu
# @File    : unit_test.py
import random
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *


SINGLE_GATE = [X, H, S, S_dagger, X, Y, Z, ID, U1, U2, U3, Rx, Ry, Rz, T, T_dagger]


def getRandomList(l, n):
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:l]


def generate_gate(gate, n, circuit):
    generator_number = gate.controls + gate.targets
    rand = getRandomList(generator_number, n)
    gate.cargs = rand[:gate.controls]
    gate.targs = rand[gate.controls:gate.controls + gate.targets]
    if gate.params > 0:
        gate.pargs = [random.random() * 2 * np.pi for _ in range(gate.params)]
    return circuit(rand[:gate.controls + gate.targets]), gate


def test_single():
    for gate in SINGLE_GATE:
        circuit = Circuit(1)
        qureg, gen_g = generate_gate(gate, 1, circuit)
        gen_g | qureg
        unitary = circuit.matrix()
        ans = np.asmatrix(gen_g.matrix)
        ans = ans.reshape(2, 2)
        if (abs(abs(unitary - ans)) > 1e-10).any():
            print(unitary, gen_g.matrix, gen_g.inverse().matrix, gen_g)
            assert 0
    assert 1
