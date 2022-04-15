import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import Ry, Rz, CX
from QuICT.algorithm.synthetical_unitary import SyntheticalUnitary
from QuICT.qcda.synthesis.unitary_transform.two_qubit_transform import CartanKAKDecomposition, TwoQubitTransform
from QuICT.qcda.synthesis.unitary_transform.two_qubit_diagonal_transform import TwoQubitDiagonalTransform


def generate_unitary(n):
    detM = 0
    while np.isclose(detM, 0, rtol=1.0e-13, atol=1.0e-13):
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        M = A + 1j * B
        detM = np.linalg.det(M)
    U, _, _ = np.linalg.svd(M)
    return U


def Ud(a, b, c):
    """
    Exp(i(a XX + b YY + c ZZ))
    """
    return np.array([[np.exp(1j * c) * np.cos(a - b), 0, 0, 1j * np.exp(1j * c) * np.sin(a - b)],
                     [0, np.exp(-1j * c) * np.cos(a + b), 1j * np.exp(-1j * c) * np.sin(a + b), 0],
                     [0, 1j * np.exp(-1j * c) * np.sin(a + b), np.exp(-1j * c) * np.cos(a + b), 0],
                     [1j * np.exp(1j * c) * np.sin(a - b), 0, 0, np.exp(1j * c) * np.cos(a - b)]], dtype=complex)


def test_tensor_decompose():
    U0 = generate_unitary(2)
    U1 = generate_unitary(2)
    U = np.kron(U0, U1)
    CartanKAKDecomposition.tensor_decompose(U)


def test_CKD():
    for _ in range(20):
        U = generate_unitary(4)
        CKD = CartanKAKDecomposition(U)
        CKD.decompose()

        # KL0 = CKD.KL0
        # KL1 = CKD.KL1
        # KR0 = CKD.KR0
        # KR1 = CKD.KR1
        # KL = np.kron(KL0, KL1)
        # KR = np.kron(KR0, KR1)
        matexp = Ud(CKD.a, CKD.b, CKD.c)

        circuit = Circuit(2)
        Rz(np.pi / 2) | circuit(1)
        CX | circuit([1, 0])
        Rz(np.pi / 2 - 2 * CKD.c) | circuit(0)
        Ry(np.pi / 2 - 2 * CKD.a) | circuit(1)
        CX | circuit([0, 1])
        Ry(2 * CKD.b - np.pi / 2) | circuit(1)
        CX | circuit([1, 0])
        Rz(-np.pi / 2) | circuit(0)

        U /= np.linalg.det(U) ** 0.25
        # Usyn = KL.dot(matexp).dot(KR)
        # print(U.dot(np.linalg.inv(Usyn)))
        Ucir = SyntheticalUnitary.run(circuit, showSU=True)

        assert np.allclose(matexp.dot(np.linalg.inv(Ucir)), np.eye(4)) \
               or np.allclose(matexp.dot(np.linalg.inv(Ucir)), 1j * np.eye(4))


def test_two_bit_transform():
    for _ in range(200):
        U = generate_unitary(4)
        circuit = Circuit(2)
        TwoQubitTransform.execute(U) | circuit

        Ucir = SyntheticalUnitary.run(circuit)
        phase = U.dot(np.linalg.inv(Ucir))
        assert np.allclose(phase, phase[0, 0] * np.eye(4))


def test_two_qubit_diagonal_transform():
    for _ in range(200):
        U = generate_unitary(4)
        U /= np.linalg.det(U) ** 0.25
        circuit = Circuit(2)
        TwoQubitDiagonalTransform.execute(U) | circuit

        Ucir = SyntheticalUnitary.run(circuit)
        phase = U.dot(np.linalg.inv(Ucir))
        assert np.allclose(phase, phase[0, 0] * np.eye(4))