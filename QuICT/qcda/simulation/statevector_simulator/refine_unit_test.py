#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/28 6:28 下午
# @Author  : Han Yu
# @File    : refine_unit_test

import numpy as np
import numba

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.simulation.statevector_simulator.refine_statevector_simulator import *
from QuICT.qcda.simulation.statevector_simulator.constant_statevecto_simulator import *
from time import time

from QuICT.ops.linalg.gpu_calculator import *


test_result = False
run_without_predata = False


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
    # pre-compiled kernel function
    print("Pre-compiled.")
    qubit_numt = 10
    circuitt = Circuit(qubit_numt)
    QFT.build_gate(qubit_numt) | circuitt

    _ = ConstantStateVectorSimulator.run_predata(circuitt)

    for i in range(1):
        print("Start running.")
        qubit_num = 30
        circuit = Circuit(qubit_num)
        # circuit.random_append(500)
        # X | circuit([0])
        QFT.build_gate(qubit_num) | circuit
        #QFT.build_gate(qubit_num) | circuit
        #QFT.build_gate(qubit_num) | circuit
        # print(circuit.circuit_size())
        #QFT.build_gate(qubit_num) | circuit
        #QFT.build_gate(qubit_num) | circuit

        if run_without_predata:
            with numba.cuda.defer_cleanup():
                start_time = time()
                state = ConstantStateVectorSimulator.run(circuit)
                end_time = time()
                duration_1 = end_time - start_time
        else:
            start_time = time()
            state = ConstantStateVectorSimulator.run_predata(circuit)
            end_time = time()
            duration_1 = end_time - start_time
        
        print(f"Cur algo: {duration_1} s.")

        # Test result
        if test_result:
            start_time = time()
            state_expected = Amplitude.run(circuit) 
            end_time = time()
            duration_2 = end_time - start_time

            assert(np.allclose(state, state_expected))
            print(f"Old algo: {duration_2} s.")

test_constant_vec_sim()
