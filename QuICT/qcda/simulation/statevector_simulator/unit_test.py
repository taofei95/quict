#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py

import pytest

import numpy as np

from QuICT.algorithm import Amplitude
from QuICT.core import *
from QuICT.qcda.simulation.statevector_simulator import *

"""
the file describe Simulators between two basic gates.
"""

@pytest.mark.repeat(5)
def w_test_vector_generate():
    qubit = 10
    gate_number = 1000
    circuit = Circuit(qubit)
    circuit.random_append(gate_number, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    circuit.print_information()
    # X | circuit[1]
    # CX | circuit
    # CX | circuit([1, 0])
    circuit_amplitude = Amplitude.run(circuit)
    pp: list = np.random.permutation(gate_number - 1).tolist()
    pp.append(-1)
    print(pp)
    vector = StateVectorSimulator.act_unitary_by_ordering(circuit.gates, pp, qubit)
    assert np.allclose(circuit_amplitude, vector)

# @pytest.mark.repeat(5)
def test_vector_run():
    circuit = Circuit(10)
    circuit.random_append(1000, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    # circuit.print_information()
    circuit_amplitude = Amplitude.run(circuit)
    stateVector = StateVectorSimulator.run(circuit)
    # a = np.allclose(small_matrix, unitary)
    # b = np.allclose(small_matrix, circuit_unitary)
    # c = np.allclose(circuit_unitary, unitary)
    # print(a, b, c)
    assert np.allclose(circuit_amplitude, stateVector)
