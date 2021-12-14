#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py

import pytest

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.simulation.gpu_simulator.unitary_simulator import *

from time import time


@pytest.mark.repeat(5)
def test_merge_two_unitary_list():
    qubit = 10
    gate_number = 100
    circuit = Circuit(qubit)
    circuit.random_append(gate_number, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    circuit_unitary = SyntheticalUnitary.run(circuit)

    mat_now = circuit.gates[0].compute_matrix
    args_now = circuit.gates[0].affectArgs
    for gate in circuit.gates[1:]:
        gate: BasicGate
        mat_now, args_now = UnitarySimulator.merge_two_unitary(mat_now, args_now, gate.compute_matrix, gate.affectArgs)

    mat_now, _ = UnitarySimulator.merge_two_unitary(
        np.identity(1 << qubit, dtype=np.complex64),
        [i for i in range(qubit)],
        mat_now,
        args_now
    )
    assert np.allclose(circuit_unitary, mat_now)


@pytest.mark.repeat(5)
def test_unitary_generate():
    qubit = 10
    gate_number = 100
    circuit = Circuit(qubit)
    circuit.random_append(gate_number)

    start_time = time()
    circuit_unitary = SyntheticalUnitary.run(circuit)
    end_time = time()
    duration_1 = end_time - start_time
    start_time = time()
    sim = UnitarySimulator(circuit)
    result_mat = sim.run()
    end_time = time()
    duration_2 = end_time - start_time
    print(f"\nOld algo time: {duration_1:.4f} s, current algo time: {duration_2:.4f} s")
    assert np.allclose(circuit_unitary, result_mat)