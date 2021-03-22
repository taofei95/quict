"""
Decomposition of unitary matrix U∈SU(2^n)
"""

from .._synthesis import Synthesis
from .two_qubit_transform import KAK

class UnitaryTransform(Synthesis):
    """
    Decompose unitary matrix U∈SU(2^n) to a circuit inductively, the result 
    circuit contains only 1-qubit gates and CNOT gates.

    Step 1: Inductive decomposition (n -> n-1)
            with Cosine-Sine Decomposition
    Step 2(optional): Stop Step 1 at n = 2, 
                    use Cartan KAK Decomposition instead

    Restricted by the current research, recursive_basis can only be set to 1,2,
    other value would raise an error.

    Reference:
        arxiv.org/abs/1501.06911
        arxiv.org/abs/0806.4015
    """
    def __call__(self, matrix, recursive_basis=1, eps=1e-15):
        """
        give parameters to the UTrans

        Args:
            matrix(np.array): Unitary matrix to be decomposed
            recursive_basis(int, optional): When to stop the inductive process
            eps(float, optional): Eps of decomposition process
        """
        self.pargs = [matrix, recursive_basis, eps]
        return self


    def build_gate(self):
        """
        Return:
            circuit(Circuit): Decomposed circuit
        """
        matrix = self.pargs[0]
        recursive_basis = self.pargs[1]
        eps = self.pargs[2]
        # TODO: Cosine-Sine Decomposition goes here


UTrans = UnitaryTransform()
