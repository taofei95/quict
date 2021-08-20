#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/22 上午10:10 
# @Author  : Kaiqi Li
# @File    : unit_test.py

from math import tau
from time import time
import numpy as np

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.qcda.simulation.statevector_simulator.constant_statevector_simulator import ConstantStateVectorSimulator


"""
the file describe Simulators between two basic gates.
"""


def test_constant_statevectorsimulator():
    # pre-compiled kernel function
    print("Start running.")
    
    qubit_num = 5
    circuit = Circuit(qubit_num)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    Measure | circuit(0)
    Measure | circuit(1)
    # Perm([0,1,2,3,7,5,6,4]) | circuit
    # for i in range(500):
    #     # CZ | circuit([i%(qubit_num - 1), i%(qubit_num - 1) + 1])
    #     H | circuit(i % qubit_num)

    # ncircuit = Circuit(qubit_num)
    # QFT.build_gate(qubit_num) | ncircuit
    # QFT.build_gate(qubit_num) | ncircuit
    # QFT.build_gate(qubit_num) | ncircuit
    # Swap | ncircuit([4,7])

    s_time = time()
    simulator = ConstantStateVectorSimulator(
        circuit=circuit,
        precision=np.complex128,
        gpu_device_id=0,
        sync=True)
    state = simulator.run()
    e_time = time()
    duration_1 = e_time - s_time

    print(state)

    start_time = time()
    state_expected = Amplitude.run(circuit)

    end_time = time()
    duration_2 = end_time - start_time

    print(np.array(state_expected))

    # assert np.allclose(state.get(), state_expected)

    # print(f"Cur algo: {duration_1} s.")
    # print(f"Old algo: {duration_2} s.")


def qiskit_test():
    from qiskit import QuantumCircuit, execute
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.models import BackendConfiguration

    qubit_num = 25
    circ = QuantumCircuit(qubit_num)
    for i in range(500):
        x = i % (qubit_num -1)
        j = x + 1
        params = 2 * np.pi
        circ.h(x)

    backend = AerSimulator(method="statevector", precision="double", device="GPU")
    start = time()
    result = execute(circ, backend=backend, shots=1, optimization_level=0).result()
    end = time()

    print(result)
    print(end - start)


def get_test():
    qubit_num = 5
    circuit = Circuit(qubit_num)
    z = Perm([0,1,2,3,7,5,6,4])
    print(z.pargs)


test_constant_statevectorsimulator()
# qiskit_test()
# get_test()
