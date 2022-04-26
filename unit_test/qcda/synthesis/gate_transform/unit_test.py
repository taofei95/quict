#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 1:11 下午
# @Author  : Han Yu
# @File    : unit_test

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *


MAX_ITER = 6
GATE_NUM = 20


def test_google():
    for i in range(2, MAX_ITER):
        circuit = Circuit(i)
        circuit.random_append(GATE_NUM)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(GoogleSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert gates.equal(gates_tran)


def test_ustc():
    for i in range(2, MAX_ITER):
        circuit = Circuit(i)
        circuit.random_append(GATE_NUM)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(USTCSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert gates.equal(gates_tran)


def test_ibmq():
    for i in range(2, MAX_ITER):
        circuit = Circuit(i)
        circuit.random_append(GATE_NUM)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(IBMQSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert gates.equal(gates_tran)


def test_ionq():
    for i in range(2, MAX_ITER):
        circuit = Circuit(i)
        circuit.random_append(GATE_NUM)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(IonQSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert gates.equal(gates_tran)


def test_buildZyz():
    buildSet = InstructionSet([CY, Rz, Ry])
    for i in range(2, MAX_ITER):
        circuit = Circuit(i)
        circuit.random_append(GATE_NUM)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(buildSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert gates.equal(gates_tran)


def test_buildZyzWithRegister():
    buildSet = InstructionSet([CY, Rz, Ry])
    buildSet.register_SU2_rule(ZyzRule)
    buildSet.register_rule_map(Cx2CyRule)
    for i in range(2, MAX_ITER):
        circuit = Circuit(i)
        circuit.random_append(GATE_NUM)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(buildSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert gates.equal(gates_tran)
