import numpy as np
from scipy.stats import unitary_group

from QuICT.core import Circuit
from QuICT.core.gate.complex_gate.unitary_decomposition.cartan_kak_decomposition import CartanKAKDecomposition
from QuICT.core.gate.complex_gate.unitary_decomposition.cartan_kak_diagonal_decomposition import CartanKAKDiagonalDecomposition


def Ud(a, b, c):
    """
    Exp(i(a XX + b YY + c ZZ))
    """
    return np.array([[np.exp(1j * c) * np.cos(a - b), 0, 0, 1j * np.exp(1j * c) * np.sin(a - b)],
                     [0, np.exp(-1j * c) * np.cos(a + b), 1j * np.exp(-1j * c) * np.sin(a + b), 0],
                     [0, 1j * np.exp(-1j * c) * np.sin(a + b), np.exp(-1j * c) * np.cos(a + b), 0],
                     [1j * np.exp(1j * c) * np.sin(a - b), 0, 0, np.exp(1j * c) * np.cos(a - b)]], dtype=complex)


def test_tensor_decompose():
    U0 = unitary_group.rvs(2)
    U1 = unitary_group.rvs(2)
    U = np.kron(U0, U1)
    CartanKAKDecomposition.tensor_decompose(U)


def test_cartan_kak():
    for _ in range(10):
        U = unitary_group.rvs(4)
        circuit = Circuit(2)
        CKD = CartanKAKDecomposition()
        CKD.execute(U) | circuit

        Ucir = circuit.matrix()
        phase = U.dot(np.linalg.inv(Ucir))
        assert np.allclose(phase, phase[0, 0] * np.eye(4))


def test_cartan_kak_diagonal():
    for _ in range(10):
        U = unitary_group.rvs(4)
        U /= np.linalg.det(U) ** 0.25
        circuit = Circuit(2)
        CKDD = CartanKAKDiagonalDecomposition()
        CKDD.execute(U) | circuit

        Ucir = circuit.matrix()
        phase = U.dot(np.linalg.inv(Ucir))
        assert np.allclose(phase, phase[0, 0] * np.eye(4))
