import numpy as np

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.cpu_simulator import CircuitSimulator
from scipy.stats import unitary_group


def test_sim():
    for qubit_num in range(2, 20):
        circuit = Circuit(qubit_num)
        circuit.random_append(20)
        _ = CircuitSimulator().run(circuit)  # New simulator would be used by default.
        # expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
        # flag = np.allclose(res, expected)
        # assert flag
        # print(f"Testing for qubit {qubit_num}: {flag}")


def test_complex_gate():
    for qubit_num in range(3, 20):
        circuit = Circuit(qubit_num)
        QFT(qubit_num) | circuit
        CCX | circuit([0, 1, 2])
        CCRz(0.1) | circuit([0, 1, 2])
        _ = CircuitSimulator().run(circuit)  # New simulator would be used by default.
        # expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
        # flag = np.allclose(res, expected)
        # assert flag


def test_measure_gate():
    qubit_num = 4
    # measure_res_acc = 0
    for _ in range(30):
        circuit = Circuit(qubit_num)
        H | circuit
        simulator = CircuitSimulator()
        _ = simulator.run(circuit)
        measure_res = simulator.sample(circuit)

        # print(measure_res)


def test_unitary_gate():
    qubit_num = 4
    circuit = Circuit(qubit_num)
    Unitary(unitary_group.rvs(2)) | circuit([0])
    Unitary(unitary_group.rvs(2)) | circuit([1])
    Unitary(unitary_group.rvs(2)) | circuit([2])
    Unitary(unitary_group.rvs(2)) | circuit([3])
    Unitary(unitary_group.rvs(2 ** 2)) | circuit([0, 1])
    Unitary(unitary_group.rvs(2 ** 2)) | circuit([2, 0])
    Unitary(unitary_group.rvs(2 ** 2)) | circuit([3, 1])

