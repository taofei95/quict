#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/7 9:54 下午
# @Author  : Han Yu
# @File    : _unit_test.py

import pytest
from QuICT.models import *
from QuICT.algorithm import SyntheticalUnitary
import random
import numpy as np
from scipy.stats import ortho_group
import copy

single_gate = [X, H, S, S_dagger, X, Y, Z, ID, U1, U2, U3, Rx, Ry, Rz, T, T_dagger]
other_gate = [CZ, CX, CY, CH, CRz, CCX]


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

def generator_per(n):
    s = [i for i in range(1 << n)]
    np.random.shuffle(s)
    return s

def generator_custom(n):
    N = 1 << n
    Q = np.mat(ortho_group.rvs(dim=N))
    Q = Q.reshape(N, N)
    diag1 = []
    diag2 = []
    for i in range(N):
        ran = random.random()
        diag1.append(ran)
        diag2.append(np.sqrt(1 - ran * ran))

    d1 = np.diag(diag1)
    d2 = np.diag(diag2)
    A = Q.T * d1 * Q
    B = Q.T * d2 * Q
    U = A + B[:] * 1j
    return np.array(U).reshape(1, -1).tolist()

def test_single():
    for gate in single_gate:
        circuit = Circuit(1)
        qureg, gen_g = generate_gate(gate, 1, circuit)
        gen_g | qureg
        gen_g.inverse() | qureg
        unitary = SyntheticalUnitary.run(circuit)
        if (abs(abs(unitary - np.identity(2, dtype=np.complex))) > 1e-10).any():
            print(unitary, gen_g.matrix, gen_g.inverse().matrix, gen_g)
            assert 0
    assert 1

def test_other():
    for gate in other_gate:
        circuit = Circuit(gate.controls + gate.targets)
        qureg, gen_g = generate_gate(gate, gate.controls + gate.targets, circuit)
        gen_g | qureg
        gen_g.inverse() | qureg
        unitary = SyntheticalUnitary.run(circuit)
        if (abs(abs(unitary - np.identity((1 << gate.controls + gate.targets), dtype=np.complex))) > 1e-10).any():
            print(unitary, gen_g)
            assert 0
    assert 1

def test_perm():
    max_test = 5
    every_round = 20
    for i in range(1, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            plist = generator_per(i)
            Perm(plist) | circuit
            Perm(plist).inverse() | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                print(unitary, plist)
                assert 0
    assert 1

def test_custom():
    max_test = 5
    every_round = 20
    for i in range(1, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            plist = generator_custom(i)[0]
            Custom(plist) | circuit
            Custom(plist).inverse() | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                print(unitary, plist)
                assert 0
    assert 1

def test_circuit():
    gates = copy.deepcopy(single_gate)
    gates.extend(other_gate)
    max_test = 5
    every_round = 20
    max_step = 100
    for i in range(1, max_test + 1):
        for _ in range(1, every_round):
            circuit = Circuit(i)
            for _ in range(1, max_step):
                gate = None
                while gate is None:
                    gate = gates[random.randrange(len(gates))]
                    if gate.controls + gate.targets > i:
                        gate = None
                qureg, gen_g = generate_gate(gate, i, circuit)
                gen_g           | qureg
                gen_g.inverse() | qureg
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_qureg():
    circuit = Circuit(5)
    qureg1 = circuit([2])
    qureg2 = circuit([3])
    qureg3 = qureg1 + qureg2
    for qubit in qureg3:
        print(qubit.id)
    Measure | qureg3

if __name__ == "__main__":
    pytest.main(["./_unit_test.py"])
