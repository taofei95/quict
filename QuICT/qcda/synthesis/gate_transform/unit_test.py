#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 1:11 下午
# @Author  : Han Yu
# @File    : unit_test


import pytest

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *

def test_google():
    for i in range(2, 10):
        circuit = Circuit(i)
        circuit.random_append(100)
        compositeGate = GateTransform.execute(circuit, GoogleSet)
        B = CompositeGate(circuit, with_copy=False)
        assert compositeGate.equal(B)

def test_ustc():
    for i in range(2, 10):
        circuit = Circuit(i)
        circuit.random_append(100)
        compositeGate = GateTransform.execute(circuit)
        B = CompositeGate(circuit, with_copy=False)
        assert compositeGate.equal(B)

def test_ibmq():
    for i in range(2, 10):
        circuit = Circuit(i)
        circuit.random_append(100)
        compositeGate = GateTransform.execute(circuit, IBMQSet)
        B = CompositeGate(circuit, with_copy=False)
        assert compositeGate.equal(B)

def test_ionq():
    for i in range(2, 10):
        circuit = Circuit(i)
        circuit.random_append(100)
        compositeGate = GateTransform.execute(circuit, IonQSet)
        B = CompositeGate(circuit, with_copy=False)
        assert compositeGate.equal(B)

def test_buildZyz():
    buildSet = InstructionSet([CY, Rz, Ry])
    for i in range(2, 10):
        circuit = Circuit(i)
        circuit.random_append(100)
        compositeGate = GateTransform.execute(circuit, buildSet)
        B = CompositeGate(circuit, with_copy=False)
        assert compositeGate.equal(B)

def test_buildZyzWithRegister():
    buildSet = InstructionSet([CY, Rz, Ry])
    buildSet.register_SU2_rule(ZyzRule)
    buildSet.register_rule_map(Cx2CyRule)
    for i in range(2, 10):
        circuit = Circuit(i)
        circuit.random_append(100)
        compositeGate = GateTransform.execute(circuit, buildSet)
        B = CompositeGate(circuit, with_copy=False)
        assert compositeGate.equal(B)

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
