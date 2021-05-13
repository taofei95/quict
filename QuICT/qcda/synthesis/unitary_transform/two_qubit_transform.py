"""
Decomposition of SU(4) with Cartan KAK Decomposition
"""

from typing import *

import numpy as np

from QuICT.core import CompositeGate, CX, Ry, Rz, Unitary
from .._synthesis import Synthesis

# Magic basis
B = (1.0 / np.sqrt(2)) * np.array([[1, 1j, 0, 0],
                                   [0, 0, 1j, 1],
                                   [0, 0, 1j, -1],
                                   [1, -1j, 0, 0]], dtype=complex)


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
        [1] https://arxiv.org/abs/0806.4015
        [2] https://arxiv.org/abs/quant-ph/0507171
        [3] https://arxiv.org/abs/quant-ph/0308006
    """

    def __init__(self, matrix, eps=1e-15):
        """

        Args:
            matrix(np.array): 4*4 unitary matrix to be decomposed
            eps(float, optional): Eps of decomposition process
        """
        self.matrix = matrix
        self.eps = eps

    @staticmethod
    def diagonalize_unitary_symmetric(matrix):
        """
        Diagonalize unitary symmetric matrix with real orthogonal matrix
        TODO: this part is taken from qiskit, the FIXME should be removed.

        Args:
            matrix(np.array): unitary symmetric matrix to be diagonalized
        
        Returns:
            Tuple[np.array, np.array]: Eigenvalues and eigenvectors
        """
        M2 = matrix.copy()
        # D, P = la.eig(M2)  # this can fail for certain kinds of degeneracy
        for i in range(100):  # FIXME: this randomized algorithm is horrendous
            state = np.random.default_rng(i)
            M2real = state.normal() * M2.real + state.normal() * M2.imag
            _, P = np.linalg.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=1.0e-13, atol=1.0e-13):
                break
        else:
            raise ValueError("CartanKAKDecomposition: failed to diagonalize M2")
        return D, P

    @staticmethod
    def tensor_decompose(matrix):
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

        # Some preparation derived from Cartan involution
        Up = B.T.conj().dot(U).dot(B)
        M2 = Up.T.dot(Up)
        M2.real[abs(M2.real) < eps] = 0.0
        M2.imag[abs(M2.imag) < eps] = 0.0

        # Since M2 is a symmetric unitary matrix, we can diagonalize its real and
        # imaginary part simultaneously. That is, ∃ P∈SO(4), s.t. M2 = P.D.P^T, 
        # where D is diagonal with unit-magnitude elements.
        D, P = self.diagonalize_unitary_symmetric(M2)

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
    @classmethod
    def execute(cls, matrix, eps=1e-15):
        """
        Decompose a matrix U∈SU(4) with Cartan KAK Decomposition to 
        a circuit, which contains only 1-qubit gates and CNOT gates.
        The decomposition of Exp(i(a XX + b YY + c ZZ)) may vary a global phase.

        Args:
            matrix(np.array): 4*4 unitary matrix to be decomposed
            eps(float, optional): Eps of decomposition process
        
        Returns:
            CompositeGate: Decomposed gates.

        Reference:
            [1] https://arxiv.org/abs/0806.4015
            [2] https://arxiv.org/abs/quant-ph/0308006
        """

        assert matrix.shape == (4, 4), \
            ValueError("TwoQubitTransform: Input must be a 4*4 matrix.")
        assert np.allclose(matrix.T.conj().dot(matrix), np.eye(4)), \
            ValueError("TwoQubitTransform: Input must be a unitary matrix.")

        CKD = CartanKAKDecomposition(matrix, eps)
        CKD.decompose()

        KL0 = CKD.KL0.dot(Rz(-np.pi / 2).matrix.reshape(2, 2))
        KL1 = CKD.KL1
        KR0 = CKD.KR0
        KR1 = Rz(np.pi / 2).matrix.reshape(2, 2).dot(CKD.KR1)
        gates = CompositeGate()
        with gates:
            # @formatter:off
            Unitary(KR0)                & 0
            Unitary(KR1)                & 1
            CX                          & [1, 0]
            Rz(np.pi / 2 - 2 * CKD.c)   & 0
            Ry(np.pi / 2 - 2 * CKD.a)   & 1
            CX                          & [0, 1]
            Ry(2 * CKD.b - np.pi / 2)   & 1
            CX                          & [1, 0]
            Unitary(KL0)                & 0
            Unitary(KL1)                & 1
            # @formatter:on

        return gates
