"""
Decomposition of SU(4) with Cartan KAK Decomposition
"""

import numpy as np

from QuICT.core import Circuit, Unitary, Ry, Rz, CX
from .._synthesis import Synthesis


class CartanKAKDecomposition:
    """Cartan KAK Decomposition in SU(4)

    ∀ U∈SU(4), ∃ KL0, KL1, KR0, KR1∈SU(2), a, b, c∈ℝ, s.t.
    U = (KL0⊗KL1).exp(i(a XX + b YY + c ZZ)).(KR0⊗KR1)

    Proof of this proposition in general cases is too 'mathematical' even for TCS 
    researchers. [2] gives a proof for U(4), which is a little more friendly for 
    researchers without mathematical background. However, be aware that [2] uses 
    a different notation.

    The process of this part is taken from [1], while some notation is from [3], 
    which is useful in the build_gate.
    Reference:
        [1] arxiv.org/abs/0806.4015
        [2] arxiv.org/abs/quant-ph/0507171
        [3] arxiv.org/abs/quant-ph/0308006
    """

    def __init__(self, matrix, eps=1e-15):
        """
        Args:
            matrix(np.array): 4*4 unitary matrix to be decomposed
            eps(float, optional): Eps of decomposition process
        """
        self.matrix = matrix
        self.eps = eps

    def tensor_decompose(self, matrix):
        """
        Decompose U∈SU(2)⊗SU(2) to U0, U1∈SU(2), s.t. U = U0⊗U1

        Args:
            matrix(np.array): Matrix to be decomposed

        Returns:
            Tuple(np.array): Decomposed result U0, U1
        """
        U = matrix.copy()
        # Decompose U1
        U1 = U[:2, :2].copy()
        # There is chance that U0[0, 0] == 0 (or more explicitly, is close to 0)
        if np.abs(np.linalg.det(U1)) < 0.1:
            U1 = U[2:, :2].copy()
        # If U0[0, 0], U0[1, 0] are both close to 0, U0 would not be unitary.
        if np.abs(np.linalg.det(U1)) < 0.1:
            raise ValueError("tensor_decompose: U1 failed")
        U1 /= np.sqrt(np.linalg.det(U1))

        # Decompose U0
        U1_inv = np.kron(np.eye(2), U1.T.conj())
        U0_tensor = U1_inv.dot(U)
        U0 = U0_tensor[::2, ::2]
        if np.abs(np.linalg.det(U0)) < 0.9:
            raise ValueError("tensor_decompose: U0 failed")
        U0 /= np.sqrt(np.linalg.det(U0))

        # Final test
        res = np.kron(U0, U1)
        dev = np.abs(np.abs(res.conj(res).T.dot(U).trace()) - 4)
        assert dev < 1e-13, \
            ValueError("tensor_decompose: Final failed")
        return U0, U1

    def decompose(self):
        """
        Decomposition process
        """
        U = self.matrix.copy()
        U /= np.linalg.det(U) ** (0.25)
        eps = self.eps

        # Magic basis
        B = (1.0 / np.sqrt(2)) * np.array([[1, 1j, 0, 0],
                                           [0, 0, 1j, 1],
                                           [0, 0, 1j, -1],
                                           [1, -1j, 0, 0]], dtype=complex)

        # Some preparation derived from Cartan involution
        Up = B.T.conj().dot(U).dot(B)
        M2 = Up.T.dot(Up)
        M2.real[abs(M2.real) < eps] = 0.0
        M2.imag[abs(M2.imag) < eps] = 0.0

        # Since M2 is a symmetric unitary matrix, we can diagonalize its real and
        # imaginary part simultaneously. That is, ∃ P∈SO(4), s.t. M2 = P.D.P^T, 
        # where D is diagonal with unit-magnitude elements.
        # TODO: this part is taken from qiskit, the FIXME should be removed.
        # D, P = la.eig(M2)  # this can fail for certain kinds of degeneracy
        for i in range(100):  # FIXME: this randomized algorithm is horrendous
            state = np.random.default_rng(i)
            M2real = state.normal() * M2.real + state.normal() * M2.imag
            _, P = np.linalg.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=1.0e-13, atol=1.0e-13):
                break
        else:
            raise ValueError("TwoQubitWeylDecomposition: failed to diagonalize M2")

        # Calculated D is usually in U(4) instead of SU(4), therefore d[3] is reset 
        # so that D is now in SU(4)
        d = np.angle(D) / 2
        d[3] = -d[0] - d[1] - d[2]
        self.a = (d[0] + d[2]) / 2
        self.b = (d[1] + d[2]) / 2
        self.c = (d[0] + d[1]) / 2

        # Similarly, P could be in O(4) instead of SO(4).
        if np.linalg.det(P) < 0:
            P[:, -1] = -P[:, -1]

        # Now is the time to calculate KL and KR
        KL = B.dot(Up).dot(P).dot(np.diag(np.exp(-1j * d))).dot(B.T.conj())
        KR = B.dot(P.T).dot(B.T.conj())
        KL.real[abs(KL.real) < eps] = 0.0
        KL.imag[abs(KL.imag) < eps] = 0.0
        KR.real[abs(KR.real) < eps] = 0.0
        KR.imag[abs(KR.imag) < eps] = 0.0
        self.KL0, self.KL1 = self.tensor_decompose(KL)
        self.KR0, self.KR1 = self.tensor_decompose(KR)


class TwoQubitTransform(Synthesis):
    """
    Decompose a matrix U∈SU(4) with Cartan KAK Decomposition to 
    a circuit, which contains only 1-qubit gates and CNOT gates.

    Reference:
        arxiv.org/abs/0806.4015
        arxiv.org/abs/quant-ph/0308006
    """

    def __call__(self, matrix, eps=1e-15):
        """
        give parameters to the KAK

        Args:
            matrix(np.array): 4*4 unitary matrix to be decomposed
            eps(float, optional): Eps of decomposition process
        """
        assert matrix.shape == (4, 4), \
            ValueError("TwoQubitTransform: Input must be a 4*4 matrix.")
        assert np.allclose(matrix.T.conj().dot(matrix), np.eye(4)), \
            ValueError("TwoQubitTransform: Input must be a unitary matrix.")
        self.pargs = [matrix, eps]
        return self

    def build_gate(self):
        """
        Final process after the Cartan KAK Decomposition, which is taken from [1].
        The decomposition of Exp(i(a XX + b YY + c ZZ)) may vary a global phase.

        Reference:
            [1] arxiv.org/abs/quant-ph/0308006

        Returns:
            Tuple(gates): Decomposed gates
        """
        matrix = self.pargs[0]
        eps = self.pargs[1]

        CKD = CartanKAKDecomposition(matrix, eps)
        CKD.decompose()

        KL0 = CKD.KL0.dot(Rz(-np.pi / 2).matrix.reshape(2, 2))
        KL1 = CKD.KL1
        KR0 = CKD.KR0
        KR1 = Rz(np.pi / 2).matrix.reshape(2, 2).dot(CKD.KR1)
        circuit = Circuit(2)
        # @formatter:off
        Unitary(list(KR0.flatten())) | circuit(0)
        Unitary(list(KR1.flatten())) | circuit(1)
        CX                           | circuit([1, 0])
        Rz(np.pi / 2 - 2 * CKD.c)    | circuit(0)
        Ry(np.pi / 2 - 2 * CKD.a)    | circuit(1)
        CX                           | circuit([0, 1])
        Ry(2 * CKD.b - np.pi / 2)    | circuit(1)
        CX                           | circuit([1, 0])
        Unitary(list(KL0.flatten())) | circuit(0)
        Unitary(list(KL1.flatten())) | circuit(1)
        # @formatter:on

        return circuit


KAK = TwoQubitTransform()
