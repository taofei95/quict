"""
Decomposition of SU(4) with Cartan KAK Decomposition
"""

from .._synthesis import Synthesis

class CartanKAKDecomposition:
    """Cartan KAK Decomposition in SU(4)
    ∀ U∈SU(4), ∃ KL0, KL1, KR0, KR1∈SU(2), a, b, c∈ℝ, s.t.
    U = (KL0⊗KL1).exp(i(a XX + b YY + c ZZ)).(KR0⊗KR1)

    Proof of this proposition in general cases is too 'mathematical' even for TCS researchers.
    [2] gives a proof for U(4), which is a little more friendly for researchers without 
    mathematical background. However, be aware that [2] uses a different notation.

    The process of this part is taken from [1].
    Reference:
        [1] arxiv.org/abs/0806.4015
        [2] arxiv.org/abs/quant-ph/0507171
    """


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
            matrix(np.array): Unitary matrix to be decomposed
            eps(float, optional): Eps of decomposition process
        """
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


KAK = TwoQubitTransform()
