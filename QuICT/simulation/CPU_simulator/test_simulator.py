import numpy as np

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.simulation.CPU_simulator import CircuitSimulator


def test_sim():
    for qubit_num in range(2, 20):
        circuit = Circuit(qubit_num)
        circuit.random_append(20)
        res = Amplitude.run(circuit)  # New simulator would be used by default.
        # expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
        # flag = np.allclose(res, expected)
        # assert flag
        # print(f"Testing for qubit {qubit_num}: {flag}")


def test_complex_gate():
    for qubit_num in range(3, 20):
        circuit = Circuit(qubit_num)
        QFT | circuit
        CCX | circuit
        CCRz(0.1) | circuit
        res = Amplitude.run(circuit)  # New simulator would be used by default.
        # expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
        # flag = np.allclose(res, expected)
        # assert flag


def test_measure_gate():
    for qubit_num in range(2, 20):
        circuit = Circuit(qubit_num)
        circuit.random_append(20)
        Measure | circuit
        print()
        print(circuit.gates)
        simulator = CircuitSimulator()
        res = simulator.run(circuit)
        measure = simulator.sample()
        print()
        print(measure)
