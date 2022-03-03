#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py

import pytest

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils import GateType
from QuICT.simulation.unitary_simulator import UnitarySimulator

from time import time


@pytest.mark.repeat(5)
def test_merge_two_unitary_list():
    qubit = 10
    gate_number = 100
    circuit = Circuit(qubit)
    circuit.random_append(gate_number, typelist=[GateType.cx, GateType.x])
    circuit_unitary = SyntheticalUnitary.run(circuit)
    unitary_simulator = UnitarySimulator()

    mat_now = circuit.gates[0].matrix
    args_now = circuit.gates[0].cargs + circuit.gates[0].targs
    for gate in circuit.gates[1:]:
        gate: BasicGate
        mat_now, args_now = unitary_simulator.merge_two_unitary(
            mat_now,
            args_now,
            gate.matrix,
            gate.cargs + gate.targs
        )

    mat_now, _ = unitary_simulator.merge_two_unitary(
        np.identity(1 << qubit, dtype=np.complex128),
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
    sim = UnitarySimulator()
    result_mat = sim.run(circuit)
    end_time = time()
    duration_2 = end_time - start_time
    print(f"\nOld algo time: {duration_1:.4f} s, current algo time: {duration_2:.4f} s")
    assert np.allclose(circuit_unitary, result_mat)
