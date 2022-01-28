#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 9:04
# @Author  : Li Kaiqi
# @File    : gate_builder_unit_test.py
import pytest
import random

from QuICT.core import Qureg, Qubit
from QuICT.core.gate import build_gate, build_random_gate
from QuICT.core.utils import GateType


typelist_1qubit = [GateType.rx, GateType.ry, GateType.rz]
typelist_2qubit = [
    GateType.cx, GateType.cy, GateType.crz,
    GateType.ch, GateType.cz, GateType.Rxx,
    GateType.Ryy, GateType.Rzz, GateType.fsim
]


def test_build_gate():
    """ test for build gate function """
    for _ in range(10):
        # build 1qubit gate
        gate_type = typelist_1qubit[random.randint(0, len(typelist_1qubit) - 1)]
        q1 = Qureg(1)
        params = [random.random()]
        g1 = build_gate(gate_type, q1, params)
        assert g1.type == gate_type and g1.assigned_qubits == q1

        # build 2qubits gate
        gate_type = typelist_2qubit[random.randint(0, len(typelist_2qubit) - 1)]
        q2 = Qureg(2)
        g2 = build_gate(gate_type, q2)
        assert g2.type == gate_type and g2.assigned_qubits == q2

        # build 2qubits gate with params
        gate_type = GateType.cu3
        q3 = Qureg(2)
        params = [1, 1, 1]
        g3 = build_gate(gate_type, q3, params)
        assert g3.pargs == params and g3.assigned_qubits == q3

        # build unitary gate
        from scipy.stats import unitary_group

        gate_type = GateType.unitary
        matrix = unitary_group.rvs(2 ** 3)
        g4 = build_gate(gate_type, [1, 2, 3], matrix)
        assert g4.matrix.shape == (8, 8)

        # build special gate
        gate_type = GateType.measure
        q5 = Qubit()
        g5 = build_gate(gate_type, q5)
        assert g5.assigned_qubits[0] == q5


def test_build_random_gate():
    for _ in range(10):
        # build random 1qubit gate
        gate_type = typelist_1qubit[random.randint(0, len(typelist_1qubit) - 1)]
        rg1 = build_random_gate(gate_type, 10, random_params=True)
        assert rg1.type == gate_type

        # build random 2qubits gate
        gate_type = typelist_2qubit[random.randint(0, len(typelist_2qubit) - 1)]
        rg2 = build_random_gate(gate_type, 10, random_params=True)
        assert rg2.type == gate_type


if __name__ == "__main__":
    pytest.main(["./gate_builder_unit_test.py"])