#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Li Kaiqi
# @File    : unitary_simulator_unit_test.py

import pytest

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.unitary_simulator import UnitarySimulator


def test_unitary_generate():
    qubit = 5
    gate_number = 100
    circuit = Circuit(qubit)
    circuit.random_append(gate_number)

    circuit_unitary = SyntheticalUnitary.run(circuit)
    sim = UnitarySimulator()
    result_mat = sim.get_unitary_matrix(circuit)
    assert np.allclose(circuit_unitary, result_mat)

    _ = sim.run(circuit)
    assert 1
