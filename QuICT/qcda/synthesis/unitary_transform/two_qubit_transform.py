"""
Decomposition of SU(4) with Cartan KAK Decomposition
"""

import numpy as np

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
        pass


    def decompose(self):
        """
        Decomposition process
        """
        U = self.matrix.copy()
        eps = self.eps

        # Magic basis
        B = (1.0/np.sqrt(2)) * np.array([[1, 1j, 0, 0],
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
            M2real = state.normal()*M2.real + state.normal()*M2.imag
            _, P = np.linalg.eigh(M2real)
            D = P.T.dot(M2).dot(P).diagonal()
            if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=1.0e-13, atol=1.0e-13):
                break  
        else:
            raise ValueError("TwoQubitWeylDecomposition: failed to diagonalize M2")

        # Calculated D is usually in U(4) instead of SU(4), therefore d[3] is reset 
        # so that D is now in SU(4)
        d = np.angle(D) / 2
        d[3] = -d[0]-d[1]-d[2]
        self.a = (d[0] + d[2]) / 2
        self.b = (d[1] + d[2]) / 2
        self.c = (d[0] + d[1]) / 2

        # Similarly, P could be in O(4) instead of SO(4).
        if np.linalg.det(P) < 0:
            P[:, -1] = -P[:, -1]

        # Now is the time to calculate KL and KR
        KL = B.dot(Up).dot(P).dot(np.diag(np.exp(-1j * d))).dot(B.T.conj())
        KR = B.dot(P.T).dot(B.T.conj())
        self.KL1, self.KL2 = self.tensor_decompose(KL)
        self.KR1, self.KR2 = self.tensor_decompose(KR)


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
        assert matrix.shape() == (4, 4), \
            ValueError("TwoQubitTransform: Input must be a 4*4 matrix.")
        assert np.allclose(matrix.T.conj().dot(matrix), np.eye(4)), \
            ValueError("TwoQubitTransform: Input must be a unitary matrix.")
        self.pargs = [matrix, eps]
        return self


    def build_gate(self):
        """
        Return:
            Tuple(gates): Decomposed gates
        """
        matrix = self.pargs[0]
        eps = self.pargs[1]
        # TODO: Cartan KAK Decomposition goes here


two_qubit_transform = TwoQubitTransform()
