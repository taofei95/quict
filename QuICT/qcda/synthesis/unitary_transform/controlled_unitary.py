import numpy as np

from typing import *
from .._synthesis import Synthesis
from QuICT.core import BasicGate
from ..uniformly_gate import uniformlyRz


class QuantumShannonDecompose:
    @classmethod
    def decompose(
            cls,
            u1: np.ndarray,
            u2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose a block diagonal even-size unitary matrix.
        block_diag(u1,u2) == block_diag(v, v) @ block_diag(d, d_dagger) @ block_diag(w, w)

        Args:
            u1 (np.ndarray): upper-left block
            u2 (np.ndarray): right-bottom block

        Returns:

        """
        s = u1 @ u2.conj().T

        eig_values, v = np.linalg.eig(s)
        v_dagger = v.conj().T
        d = np.sqrt(np.diag(eig_values))

        # u1 @ u2_dagger == v @ d_square @ v_dagger

        w = d @ v_dagger @ u2

        return v, d, w


class ControlledUnitary(Synthesis):
    """Decompose a 1-bit-controlled generic unitary with multiple targets.
    """

    def __call__(
            self,
            u1: np.ndarray,
            u2: np.ndarray
    ) -> "ControlledUnitary":
        """
        Build a parameterized model.


        Args:
            u1 (np.ndarray): Operation when control bit is |0>
            u2 (np.ndarray): Operation when control bit is |1>

        Returns:
            ControlledUnitary: Model filled with parameters.
        """
        self.pargs = [u1, u2]
        self.targets = int(round(np.log2(u1.shape[0])))

        return self

    def _i_tensor_unitary(
            self,
            u: np.ndarray,
            mapping: Sequence[int] = None
    ) -> Tuple[BasicGate]:
        """
        Transform (I_{2x2} tensor U) into gates. The 1st bit is under the
        identity transform before ordering.

        Args:
            u (np.ndarray): A unitary matrix.
            mapping(Sequence[int]): Qubit ordering.

        Returns:
            Tuple[BasicGate]: Synthesized gates.
        """

        # Dynamic import to avoid circular imports
        from .unitary_transform import UTrans

        n = int(round(np.log2(u.shape[0])))
        gates = UTrans(u).build_gate(mapping=[mapping[i + 1] for i in range(n)])
        return gates

    def build_gate(
            self,
            mapping: Sequence[int] = None
    ) -> Sequence[BasicGate]:
        """
        Build gates from parameterized model according to given mapping.

        References:
            https://arxiv.org/abs/quant-ph/0406176 Theorem 12

        Args:
            mapping (Sequence[int]): Qubit ordering.

        Returns:
            Sequence[BasicGate]: Synthesized gates.
        """
        u1: np.ndarray = self.pargs[0]
        u2: np.ndarray = self.pargs[1]
        n = 1 + int(round(np.log2(u1.shape[0])))
        """qubit number
        """

        if mapping is None:
            mapping = [i for i in range(n)]

        v, d, w = QuantumShannonDecompose.decompose(u1, u2)
        gates = []

        # diag(u1, u2) == diag(v, v) @ diag(d, d_dagger) @ diag(w, w)

        # diag(w, w)
        gates.extend(self._i_tensor_unitary(w, mapping=mapping))

        # diag(d, d_dagger)
        angle_list = []
        for i in range(d.shape[0]):
            s = d[i, i]
            theta = -2 * np.log(s) / 1j
            angle_list.append(theta)

        reversed_rz = uniformlyRz(angle_list=angle_list) \
            .build_gate(mapping=[mapping[(i + 1) % n] for i in range(n)])

        gates.extend(reversed_rz)

        # diag(v, v)
        gates.extend(self._i_tensor_unitary(v, mapping=mapping))

        return gates


CUTrans = ControlledUnitary()
