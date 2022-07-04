import numpy as np
from QuICT.core import Circuit

from QuICT.quantum_state_preparation import QuantumStatePreparation
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def random_unit_vector(n):
    real = np.random.random(1 << n)
    imag = np.random.random(1 << n)
    state_vector = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)
    return state_vector


def test_with_uniformly_gates():
    for n in range(2, 6):
        for _ in range(100):
            state_vector = random_unit_vector(n)
            QSP = QuantumStatePreparation('uniformly_gates')
            gates = QSP.execute(state_vector)
            circuit = Circuit(n)
            circuit.extend(gates)
            simulator = ConstantStateVectorSimulator()
            state = simulator.run(circuit)
            assert np.allclose(state_vector, state)


def test_with_unitary_decomposition():
    for n in range(2, 6):
        for _ in range(100):
            state_vector = random_unit_vector(n)
            QSP = QuantumStatePreparation('unitary_decomposition')
            gates = QSP.execute(state_vector)
            circuit = Circuit(n)
            circuit.extend(gates)
            simulator = ConstantStateVectorSimulator()
            state = simulator.run(circuit)
            assert np.allclose(state_vector, state)
