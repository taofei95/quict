"""
Decomposition of SU(4) with Cartan KAK Decomposition
"""

from .._synthesis import Synthesis

class TwoQubitTransform(Synthesis):
    """
    Decompose a matrix U∈SU(4) with Cartan KAK Decomposition to 
    a circuit, which contains only 1-qubit gates and CNOT gates.

    Reference:
        arxiv.org/abs/0806.4015
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
            circuit(Circuit): Decomposed circuit
        """
        matrix = self.pargs[0]
        eps = self.pargs[1]
        # TODO: Cartan KAK Decomposition goes here


KAK = TwoQubitTransform()
