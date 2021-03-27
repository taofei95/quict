"""
Decomposition of unitary matrix U∈SU(2^n)
"""

import numpy as np

# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from typing import *
from QuICT.core import *
from .._synthesis import Synthesis
from .two_qubit_transform import KAK
from ..uniformly_gate import uniformlyRy


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

    def __call__(self, mat, recursive_basis=1, eps=1e-15):
        """
        give parameters to the UTrans

        Args:
            mat(np.array): Unitary matrix to be decomposed
            recursive_basis(int, optional): When to stop the inductive process
            eps(float, optional): Eps of decomposition process
        """
        if recursive_basis <= 0 or recursive_basis >= 3:
            raise NotImplementedError("Recursive must stops at 1 or 2!")
        self.pargs = [mat, recursive_basis, eps]
        return self

    def build_gate(self, mapping=None) -> Sequence[BasicGate]:
        """
        Return:
            Tuple[BasicGate]: Decomposed gates
        """

        # Dynamic import to avoid circular imports
        from .controlled_unitary import CUTrans

        mat: np.ndarray = np.array(self.pargs[0])
        recursive_basis: int = self.pargs[1]
        eps: float = self.pargs[2]
        mat_size: int = mat.shape[0]
        qubit_num = int(round(np.log2(mat_size)))

        if mapping is None:
            mapping = [i for i in range(qubit_num)]

        if recursive_basis == 2:
            raise NotImplementedError("SU(4) special decompose not implemented.")

        if qubit_num == 1:
            GateBuilder.setGateType(GATE_ID["Unitary"])
            # TODO: Unitary Gate matrix type restrictions
            parg = np.reshape(mat, -1).tolist()
            GateBuilder.setPargs(parg)
            GateBuilder.setTargs([0])
            u = GateBuilder.getGate()
            return [u]

        gates = []

        u, cs, v_dagger = cossin(mat, mat_size // 2, mat_size // 2)

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

        # v_dagger
        v1_dagger = v_dagger[:mat_size // 2, :mat_size // 2]
        v2_dagger = v_dagger[mat_size // 2:, mat_size // 2:]

        gates.extend(CUTrans(v1_dagger, v2_dagger).build_gate(mapping=mapping))

        # (c,s\\s,c)

        angle_list = []
        for i in range(mat_size // 2):
            c = mat[i, i]
            s = -mat[i, i + mat_size // 2]
            theta = np.arccos(c)
            if np.isclose(-np.sin(theta), s):
                theta = - theta
            angle_list.append(theta * 2)
        reversed_ry = uniformlyRy(angle_list=angle_list) \
            .build_gate(mapping=[mapping[(i + 1) % qubit_num] for i in range(qubit_num)])
        gates.extend(reversed_ry)

        # u
        u1 = u[:mat_size // 2, :mat_size // 2]
        u2 = u[mat_size // 2:, mat_size // 2:]

        gates.extend(CUTrans(u1, u2).build_gate(mapping=mapping))

        return gates


UTrans = UnitaryTransform()
