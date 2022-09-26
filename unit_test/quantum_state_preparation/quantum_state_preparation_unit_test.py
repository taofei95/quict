import numpy as np
from QuICT.core import Circuit

from QuICT.quantum_state_preparation import QuantumStatePreparation, SparseQuantumStatePreparation
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


def random_unit_vector(n):
    real = np.random.random(n)
    imag = np.random.random(n)
    state_vector = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)
    return state_vector


def test_with_uniformly_gates():
    for n in range(2, 6):
        for _ in range(10):
            state_vector = random_unit_vector(1 << n)
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
            state_vector = random_unit_vector(1 << n)
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
            state_vector = random_unit_vector(2)
            alpha, beta = state_vector
            gates = SparseQuantumStatePreparation.multicontrol_G(n, alpha, beta)
            omega = 2 * np.arcsin(np.abs(alpha))
            gamma = np.angle(alpha) - np.angle(beta)
            mat = np.array([
                [np.sin(omega / 2), np.exp(1j * gamma) * np.cos(omega / 2)],
                [np.exp(-1j * gamma) * np.cos(omega / 2), -np.sin(omega / 2)],
            ])
            assert np.allclose(gates.matrix()[-2:, -2:], mat)
            assert(np.isclose(gates.matrix()[-2:, -2:].dot(state_vector.T)[1], 0))


def test_reduce_state():
    simulator = ConstantStateVectorSimulator()
    sparseQSP = SparseQuantumStatePreparation()
    for n in range(2, 6):
        for k in range(2, 1 << (n - 1)):
            state_vector = np.zeros(1 << n, dtype=complex)
            nonzeros = random_unit_vector(k)
            qubits = np.random.choice(range(1 << n), k, replace=False)
            state_vector[qubits] = nonzeros
            state, width = sparseQSP.statevector_to_dict(state_vector)

            gates = sparseQSP.reduce_state(state, width)
            cir = Circuit(n)
            cir.extend(gates)

            reduced = simulator.run(cir, state_vector)
            reduced_state, _ = sparseQSP.statevector_to_dict(reduced)
            # np.set_printoptions(precision=3, suppress=True)
            assert len(reduced_state) < len(state)


def test_sparse_qsp():
    simulator = ConstantStateVectorSimulator()
    sparseQSP = SparseQuantumStatePreparation('state_vector')
    for n in range(2, 6):
        for _ in range(10):
            state_vector = random_unit_vector(1 << n)
            gates = sparseQSP.execute(state_vector)
            circuit = Circuit(n)
            circuit.extend(gates)
            simulator = ConstantStateVectorSimulator()
            state = simulator.run(circuit).get()
            phase = state_vector[0] / state[0]
            assert np.allclose(state_vector, phase * state)
