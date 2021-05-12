#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py

import pytest

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.simulation.unitary_simulator import *

"""
the file describe Simulators between two basic gates.
"""


def w_test_merge_two_unitary():
    targs = [0, 1]
    compositeGate1 = CZ & targs
    compositeGate2 = X & targs[1]
    print(UnitarySimulator.merge_two_unitary(compositeGate1, compositeGate2).compute_matrix)
    print("\nfinish\n")

@pytest.mark.repeat(5)
def w_test_merge_two_unitary_list():
    qubit = 5
    gate_number = 100
    circuit = Circuit(qubit)
    circuit.random_append(gate_number, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    circuit_unitary = SyntheticalUnitary.run(circuit)

    gate_now = circuit.gates[0]
    for gate in circuit.gates[1:]:
        gate_now = UnitarySimulator.merge_two_unitary(gate_now, gate)
    unitary = Unitary(np.identity(1 << qubit, dtype=np.complex128)) & [i for i in range(qubit)]
    gate_now = UnitarySimulator.merge_two_unitary(unitary, gate_now)
    assert np.allclose(circuit_unitary, gate_now.matrix)

@pytest.mark.repeat(5)
def test_unitary_generate():
    qubit = 10
    gate_number = 2000
    circuit = Circuit(qubit)
    circuit.random_append(gate_number, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    # CX | circuit
    # X | circuit
    circuit_unitary = SyntheticalUnitary.run(circuit)
    gate_now = UnitarySimulator.merge_unitary_by_ordering(circuit.gates,
                                                          np.random.permutation(gate_number - 1).tolist())
    unitary = Unitary(np.identity(1 << qubit, dtype=np.complex128)) & [i for i in range(qubit)]
    gate_now = UnitarySimulator.merge_two_unitary(unitary, gate_now)
    assert np.allclose(circuit_unitary, gate_now.matrix)

# @pytest.mark.repeat(5)
def w_test_unitary_run():
    circuit = Circuit(10)
    circuit.random_append(100, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    # circuit.print_information()
    circuit_unitary = SyntheticalUnitary.run(circuit)
    unitary = UnitarySimulator.run(circuit)
    # a = np.allclose(small_matrix, unitary)
    # b = np.allclose(small_matrix, circuit_unitary)
    # c = np.allclose(circuit_unitary, unitary)
    # print(a, b, c)
    assert np.allclose(circuit_unitary, unitary)
