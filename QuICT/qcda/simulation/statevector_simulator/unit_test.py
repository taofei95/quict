#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py


from time import time
import numpy as np

from QuICT.core import *
from QuICT.algorithm import Amplitude
from .constant_statevecto_simulator import ConstantStateVectorSimulator


"""
the file describe Simulators between two basic gates.
"""


def test_constant_statevectorsimulator():
    # pre-compiled kernel function
    qubit_numt = 5
    circuitt = Circuit(qubit_numt)
    QFT.build_gate(qubit_numt) | circuitt

    test = ConstantStateVectorSimulator(circuitt, np.complex64, sync=True)
    _ = test.run()

    print("Start running.")
    
    qubit_num = 20
    circuit = Circuit(qubit_num)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit

    s_time = time()
    simulator = ConstantStateVectorSimulator(circuit, np.complex64, sync=True)
    state = simulator.run()
    e_time = time()
    duration_1 = e_time - s_time

    start_time = time()
    state_expected = Amplitude.run(circuit)
    end_time = time()
    duration_2 = end_time - start_time

    assert np.allclose(state.get(), state_expected)
    print()
    print(f"Cur algo: {duration_1} s.")
    print(f"Old algo: {duration_2} s.")


test_constant_statevectorsimulator()
