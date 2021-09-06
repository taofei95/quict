#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/22 上午10:10 
# @Author  : Kaiqi Li
# @File    : unit_test.py

from time import time
import numpy as np
import random

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.algorithm import StandardGrover
from QuICT.qcda.simulation.statevector_simulator.constant_statevector_simulator import ConstantStateVectorSimulator


"""
the file describe Simulators between two basic gates.
"""

def main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)

def test_constant_statevectorsimulator_grover():
    # pre-compiled kernel function
    print("Start running.")

    qubit_num = 10
    print("[%2d-bit standard grover]..." % qubit_num)

    f = [0] * (2**qubit_num)
    target = random.randrange(0,2**qubit_num)
    f[target] = 1
    circuit = StandardGrover.build_gate(f, qubit_num, main_oracle)

    s_time = time()
    simulator = ConstantStateVectorSimulator(
        circuit=circuit,
        precision=np.complex128,
        gpu_device_id=0,
        sync=True)
    state = simulator.run()
    e_time = time()
    duration_1 = e_time - s_time

    print(f"Cur algo: {duration_1} s.")

    start_time = time()
    state_expected = Amplitude.run(circuit)
    end_time = time()
    duration_2 = end_time - start_time

    print(f"Old algo: {duration_2} s.")

    assert np.allclose(state.get(), state_expected)

def test_constant_statevectorsimulator_QFT():
    # pre-compiled kernel function
    print("Start running.")
    
    qubit_num = 25
    print("[%2d-bit QFT]..." % qubit_num)
    circuit = Circuit(qubit_num)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit

    s_time = time()
    simulator = ConstantStateVectorSimulator(
        circuit=circuit,
        precision=np.complex128,
        gpu_device_id=0,
        sync=True)
    state = simulator.run()
    e_time = time()
    duration_1 = e_time - s_time

    start_time = time()
    state_expected = Amplitude.run(circuit)
    end_time = time()
    duration_2 = end_time - start_time

    assert np.allclose(state.get(), state_expected)

    print(f"Cur algo: {duration_1} s.")
    print(f"Old algo: {duration_2} s.")


# test_constant_statevectorsimulator_QFT()
test_constant_statevectorsimulator_grover()