import numpy as np

from QuICT.qcda.synthesis.quantum_state_preparation.utility import schmidt_decompose


def random_unit_vector(n):
    real = np.random.random(n)
    imag = np.random.random(n)
    state_vector = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)
    return state_vector


def test_schmidt_decompose():
    for n in range(3, 6):
        A_qubits = 2
        state_vector = random_unit_vector(1 << n)
        l, iA, iB = schmidt_decompose(state_vector, A_qubits)
        res = np.zeros(1 << n, dtype=np.complex128)
        for i in range(len(l)):
            res += l[i] * np.kron(iA[i], iB[i])
        assert np.allclose(state_vector, res)
