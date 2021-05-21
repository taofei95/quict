#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py

import pytest

import numpy as np

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.simulation.statevector_simulator import *
from time import time

from .statevector_simulator import StateVectorSimulatorRefine
from QuICT.ops.linalg.gpu_calculator import *

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
    circuit.random_append(100, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    # circuit.print_information()
    circuit_amplitude = Amplitude.run(circuit)
    stateVector = StateVectorSimulator.run(circuit)
    # a = np.allclose(small_matrix, unitary)
    # b = np.allclose(small_matrix, circuit_unitary)
    # c = np.allclose(circuit_unitary, unitary)
    # print(a, b, c)
    assert np.allclose(circuit_amplitude, stateVector)


# @pytest.mark.repeat(20)
# def test_mat_vec_reshape():
#     vec_len = 100
#     mat_sz = 10
#     mat = np.random.rand(mat_sz, mat_sz)
#     vec = np.random.rand(vec_len)
#     result_expected = np.empty(shape=vec_len, dtype=np.float)
#     for i in range(vec_len // mat_sz):
#         result_expected[i * mat_sz:(i + 1) * mat_sz] = np.dot(mat, vec[i * mat_sz:(i + 1) * mat_sz])
#
#     result_actual = np.dot(mat, vec.reshape((mat_sz, vec_len // mat_sz), order='F'))
#     result_actual = result_actual.reshape(vec_len, order='F')
#     assert np.allclose(result_actual, result_expected)

# def test_vec_perm():
#     vec = np.arange(8)
#     per = np.array([2, 0, 1])
#     vec = VectorPermutation(vec, per, True)
#     print()
#     print(vec)

def test_vec_inv_perm():
    vec = np.arange(8)
    per = np.array([1, 2, 0])
    per_inv = np.empty_like(per)
    for i in range(len(per_inv)):
        per_inv[per[i]] = i

    vec_per = VectorPermutation(vec, per)
    vec_per_inv = VectorPermutation(vec_per, per_inv)
    assert np.allclose(vec, vec_per_inv)


@pytest.mark.repeat(20)
def test_small_mat_large_vec():
    # Cannot run successfully
    qubit_num = 10
    circuit = Circuit(qubit_num)
    circuit.random_append(1)
    # Rx(np.pi / 2) | circuit(0)
    gate: BasicGate = circuit.gates[0]

    initial_state = np.zeros(1 << qubit_num, dtype=np.complex64)
    initial_state[0] = 1.0 + 0.0j
    final_state_expected = np.dot(SyntheticalUnitary.run(circuit), initial_state)
    final_state = vectordot(gate.compute_matrix, initial_state, np.array(gate.affectArgs))
    final_state_refined = vector_dot_refined(gate.compute_matrix, initial_state, np.array(gate.affectArgs))

    print()
    print(gate.name)
    print(gate.affectArgs)
    print(gate.compute_matrix)
    print(final_state_expected)
    print(final_state_refined)

    assert np.allclose(final_state, final_state_expected) # error!
    assert np.allclose(final_state_refined, final_state_expected)


def test_refine_vec_sim():
    qubit_num = 20
    circuit = Circuit(qubit_num)
    circuit.random_append(2000)
    initial_state = np.zeros(1 << qubit_num, dtype=np.complex128)
    initial_state[0] = 1.0 + 0.0j
    start_time = time()
    state = StateVectorSimulatorRefine.run(circuit, initial_state)
    end_time = time()
    duration_1 = end_time - start_time
    start_time = time()
    state_expected = Amplitude.run(circuit)
    end_time = time()
    duration_2 = end_time - start_time
    assert np.allclose(state, state_expected)
    print()
    print(f"Cur algo: {duration_1} s.")
    print(f"Old algo: {duration_2} s.")
