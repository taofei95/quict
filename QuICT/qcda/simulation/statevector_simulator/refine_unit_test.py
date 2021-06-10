#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/28 6:28 下午
# @Author  : Han Yu
# @File    : refine_unit_test

import pytest

import numpy as np
import numba

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.simulation.statevector_simulator.refine_statevector_simulator import *
from QuICT.qcda.simulation.statevector_simulator.constant_statevecto_simulator import *
from time import time

from QuICT.ops.linalg.gpu_calculator import *

def wtest_refine_vec_sim():
    qubit_num = 30
    circuit = Circuit(qubit_num)
    # circuit.random_append(500)
    # X | circuit([0])
    QFT.build_gate(qubit_num) | circuit

    # start_time = time()
    # state_expected = Amplitude.run(circuit)
    # end_time = time()
    # duration_2 = end_time - start_time

    with numba.cuda.defer_cleanup():
        start_time = time()
        state = RefineStateVectorSimulator.run(circuit)
        end_time = time()
        duration_1 = end_time - start_time

    # print(state_expected)
    # print(state)

    # assert np.allclose(state, state_expected)
    # print()
    print(f"Cur algo: {duration_1} s.")
    # print(f"Old algo: {duration_2} s.")
    # assert 0

def test_constant_vec_sim():
    for i in range(1):
        print("Pre-compiled.")
        qubit_num = 10
        circuit = Circuit(qubit_num)
        QFT.build_gate(qubit_num) | circuit

        _ = ConstantStateVectorSimulator.run_predata_ot(circuit)

    for i in range(1):
        print("Start running.")
        qubit_num = 30
        circuit = Circuit(qubit_num)
        # circuit.random_append(500)
        # X | circuit([0])
        # QFT.build_gate(qubit_num) | circuit
        # QFT.build_gate(qubit_num) | circuit
        # QFT.build_gate(qubit_num) | circuit
        # QFT.build_gate(qubit_num) | circuit
        QFT.build_gate(qubit_num) | circuit
        # print(circuit.circuit_size())
        # QFT.build_gate(qubit_num) | circuit
        # QFT.build_gate(qubit_num) | circuit

        # start_time = time()
        # state_expected = Amplitude.run(circuit)
        # end_time = time()
        #   duration_2 = end_time - start_time
        with numba.cuda.defer_cleanup():
           start_time = time()
           state = ConstantStateVectorSimulator.run_predata_ot(circuit)
           end_time = time()
           duration_1 = end_time - start_time

        # assert np.allclose(state, state_expected)
        # print(f"Cur algo: {duration_1} s.")
        print(f"Old algo: {end_time - start_time} s.")
        # assert 0
