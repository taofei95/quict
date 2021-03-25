import sys
sys.path.append('/mnt/e/ICT/QuICT')

import numpy as np

from QuICT.core import Circuit, Unitary, Ry, Rz, CX
from QuICT.algorithm.synthetical_unitary import SyntheticalUnitary
from QuICT.qcda.synthesis.unitary_transform.two_qubit_transform import CartanKAKDecomposition, KAK

def generate_unitary(n):
    detM = 0
    while np.isclose(detM, 0, rtol=1.0e-13, atol=1.0e-13):
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        M = A + 1j * B
        detM = np.linalg.det(M)
    U, _, _ = np.linalg.svd(M)
    return U

def test_tensor_decompose():
    U0 = generate_unitary(2)
    U1 = generate_unitary(2)
    U = np.kron(U0, U1)
    CKD = CartanKAKDecomposition(U)
    CKD.tensor_decompose(U)


def test_CKD():
    for _ in range(20):
        U = generate_unitary(4)
        CKD = CartanKAKDecomposition(U)
        CKD.decompose()

        KL0 = CKD.KL0
        KL1 = CKD.KL1
        KR0 = CKD.KR0
        KR1 = CKD.KR1

        circuit = Circuit(2)
        Unitary(list(KL0.flatten())) | circuit(0)
        Unitary(list(KL1.flatten())) | circuit(1)
        Rz(np.pi / 2)                | circuit(1)
        CX                           | circuit([1, 0])
        Rz(2 * CKD.c - np.pi / 2)    | circuit(0)
        Ry(np.pi / 2 - 2 * CKD.a)    | circuit(1)
        CX                           | circuit([0, 1])
        Ry(2 * CKD.b - np.pi / 2)    | circuit(1)
        CX                           | circuit([1, 0])
        Rz(-np.pi / 2)               | circuit(0)
        Unitary(list(KR0.flatten())) | circuit(0)
        Unitary(list(KR1.flatten())) | circuit(1)

        U /= np.linalg.det(U)**(0.25)
        unitary = SyntheticalUnitary.run(circuit, showSU = True)
        print(unitary.dot(np.linalg.inv(U)))


def test():
    for _ in range(20):
        U = generate_unitary(4)
        circuit = Circuit(2)
        KAK(U) | circuit
        
        U /= np.linalg.det(U)**(0.25)
        unitary = SyntheticalUnitary.run(circuit, showSU = True)
        print(unitary.dot(np.linalg.inv(U)))


if __name__ == '__main__':
    # test_tensor_decompose()
    test_CKD()
    # test()