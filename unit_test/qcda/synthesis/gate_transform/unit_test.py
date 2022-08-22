#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 1:11 下午
# @Author  : Han Yu
# @File    : unit_test

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *


def test_google():
    for i in range(2, 6):
        circuit = Circuit(i)
        circuit.random_append(20)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(GoogleSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert np.allclose(gates.matrix(), gates_tran.matrix())


def test_ustc():
    for i in range(2, 6):
        circuit = Circuit(i)
        circuit.random_append(20)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(USTCSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert np.allclose(gates.matrix(), gates_tran.matrix())


def test_ibmq():
    for i in range(2, 6):
        circuit = Circuit(i)
        circuit.random_append(20)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(IBMQSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert np.allclose(gates.matrix(), gates_tran.matrix())


def test_ionq():
    for i in range(2, 6):
        circuit = Circuit(i)
        circuit.random_append(20)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(IonQSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert np.allclose(gates.matrix(), gates_tran.matrix())


def test_buildZyz():
    buildSet = InstructionSet(GateType.cy, [GateType.rz, GateType.ry])
    for i in range(2, 6):
        circuit = Circuit(i)
        circuit.random_append(20)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(buildSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert np.allclose(gates.matrix(), gates_tran.matrix())


def test_buildZyzWithRegister():
    buildSet = InstructionSet(GateType.cy, [GateType.rz, GateType.ry])
    buildSet.register_one_qubit_rule(zyz_rule)
    buildSet.register_two_qubit_rule_map(cx2cy_rule, GateType.cx)
    for i in range(2, 6):
        circuit = Circuit(i)
        circuit.random_append(20)
        gates = CompositeGate(gates=circuit.gates)
        GT = GateTransform(buildSet)
        circuit_tran = GT.execute(circuit)
        gates_tran = CompositeGate(gates=circuit_tran.gates)
        assert np.allclose(gates.matrix(), gates_tran.matrix())
