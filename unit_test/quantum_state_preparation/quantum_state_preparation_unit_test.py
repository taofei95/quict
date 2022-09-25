import numpy as np
from QuICT.core import Circuit

from QuICT.quantum_state_preparation import QuantumStatePreparation, SparseQuantumStatePreparation
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


def random_unit_vector(n):
    real = np.random.random(1 << n)
    imag = np.random.random(1 << n)
    state_vector = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)
    return state_vector


def test_with_uniformly_gates():
    for n in range(2, 6):
        for _ in range(10):
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
        for _ in range(10):
            state_vector = random_unit_vector(n)
            QSP = QuantumStatePreparation('unitary_decomposition')
            gates = QSP.execute(state_vector)
            circuit = Circuit(n)
            circuit.extend(gates)
            simulator = ConstantStateVectorSimulator()
            state = simulator.run(circuit)
            assert np.allclose(state_vector, state)


def test_multicontrol_G():
    for n in range(2, 6):
        for _ in range(10):
            state_vector = random_unit_vector(1)
            alpha, beta = state_vector
            gates = SparseQuantumStatePreparation.multicontrol_G(n, alpha, beta)
            omega = 2 * np.arcsin(np.abs(alpha))
            gamma = np.angle(alpha) - np.angle(beta)
            mat = np.array([
                [np.sin(omega / 2), np.exp(1j * gamma) * np.cos(omega / 2)],
                [np.exp(-1j * gamma) * np.cos(omega / 2), -np.sin(omega / 2)],
            ])
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates.matrix()[-2:, -2:], mat)
            assert(np.isclose(mat.dot(state_vector.T)[1], 0))
