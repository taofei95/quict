"""
Decomposition of unitary matrix U∈SU(2^n)
"""

import numpy as np

# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from typing import *
from QuICT.core import BasicGate
from .._synthesis import Synthesis
from .two_qubit_transform import KAK
from .controlled_unitary import CUTrans


class UnitaryTransform(Synthesis):
    """
    Decompose unitary matrix U∈SU(2^n) to a circuit inductively, the result 
    circuit contains only 1-qubit gates and CNOT gates.

    Step 1: Inductive decomposition (n -> n-1)
            with Cosine-Sine Decomposition & Quantum Shannon Decomposition.
    Step 2(optional): Stop Step 1 at n = 2, 
                    use Cartan KAK Decomposition instead.

    Restricted by the current research, recursive_basis can only be set to 1,2,
    other value would raise an error.

    Reference:
        https://arxiv.org/abs/quant-ph/0406176
        https://arxiv.org/abs/1501.06911
        https://arxiv.org/abs/0806.4015
    """

    def __call__(self, matrix, recursive_basis=1, eps=1e-15):
        """
        give parameters to the UTrans

        Args:
            matrix(np.array): Unitary matrix to be decomposed
            recursive_basis(int, optional): When to stop the inductive process
            eps(float, optional): Eps of decomposition process
        """
        if recursive_basis <= 0 or recursive_basis >= 3:
            raise NotImplementedError("Recursive must stops at 1 or 2!")
        self.pargs = [matrix, recursive_basis, eps]
        return self

    def build_gate(self, mapping=None):
        """
        Return:
            Tuple[BasicGate]: Decomposed gates
        """
        matrix: np.ndarray = self.pargs[0]
        recursive_basis: int = self.pargs[1]
        eps: float = self.pargs[2]
        mat_size: int = matrix.shape[0]
        u, cs, v_dagger = cossin(matrix, mat_size // 2, mat_size // 2)

        """
        Parts of following comments are from Scipy documentation
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cossin.html)
                                   ┌                   ┐
                                   │ I  0  0 │ 0  0  0 │
        ┌           ┐   ┌         ┐│ 0  C  0 │ 0 -S  0 │┌         ┐*
        │ X11 │ X12 │   │ U1 │    ││ 0  0  0 │ 0  0 -I ││ V1 │    │
        │ ────┼──── │ = │────┼────││─────────┼─────────││────┼────│
        │ X21 │ X22 │   │    │ U2 ││ 0  0  0 │ I  0  0 ││    │ V2 │
        └           ┘   └         ┘│ 0  S  0 │ 0  C  0 │└         ┘
                                   │ 0  0  I │ 0  0  0 │
                                   └                   ┘
                                   
        Both u and v are controlled unitary operations hence can be 
        decomposed into 2 (smaller) unitary operations and 1 controlled rotation.        
        """


UTrans = UnitaryTransform()
