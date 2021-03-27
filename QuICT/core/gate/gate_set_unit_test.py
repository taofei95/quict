#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/28 12:03 上午
# @Author  : Han Yu
# @File    : gate_set_unit_test

import pytest

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *

def test_gateSet_attributes():
    # circuit
    circuit = Circuit(5)
    circuit.random_append()
    H | circuit
    X | circuit
    gateSet = GateSet(circuit)
    assert len(gateSet.gates) == circuit.circuit_size()
    assert gateSet.circuit_width() == 5
    assert gateSet.circuit_count_1qubit() + gateSet.circuit_count_2qubit() == circuit.circuit_size()
    assert gateSet.qasm() == circuit.qasm()

    # BasicGate
    _X = X.copy()
    _X.targs = [0]
    gateSet = GateSet(_X)
    assert len(gateSet.gates) == 1
    assert gateSet.circuit_width() == 1

    # GateSet
    gateSet = GateSet(gateSet)
    assert len(gateSet.gates) == 1
    assert gateSet.circuit_width() == 1

    # list<BasicGate>
    _Y = Y.copy()
    _Y.targs = [0]
    gateSet = GateSet([_X, _Y])
    assert len(gateSet) == 2
    gateSet = GateSet((_X, _Y))
    assert len(gateSet) == 2

def test_gate_matrix():
    circuit = Circuit(5)
    circuit.random_append()
    gateSet = GateSet(circuit)
    assert np.all(gateSet.matrix() == SyntheticalUnitary.run(circuit))

    circuit2 = Circuit(6)
    gateSet | circuit2[1:]
    gateSet = GateSet(circuit2)
    assert np.all(gateSet.matrix(local=True) == SyntheticalUnitary.run(circuit))

def test_add_gate():
    gateSet = GateSet()
    with gateSet:
        CX & (0, 1)
        H & 1
    circuit = Circuit(2)
    CX | circuit
    H | circuit(1)
    Phase(0.2) | circuit
    # assert np.all(SyntheticalUnitary.run(circuit) == gateSet.matrix())
    assert gateSet.equal(circuit, ignore_phase = True)

if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    pytest.main(["./gate_set_unit_test.py"])
