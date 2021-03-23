import numpy as np

from typing import *
from .._synthesis import Synthesis
from QuICT.core import BasicGate


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
        Build a parameterized model. Both v1 and v2 might be changed during synthesis.
        So pay attention to side effects.


        Args:
            u1 (np.ndarray): Operation when control bit is |0>
            u2 (np.ndarray): Operation when control bit is |1>

        Returns:
            ControlledUnitary: Model filled with parameters.
        """
        self.pargs = [u1, u2]
        self.targets = int(round(np.log2(u1.shape[0])))

        return self

    def build_gate(
            self,
            mapping: Sequence[int] = None
    ) -> Tuple[BasicGate]:
        """
        Build gates from parameterized model according to given mapping.

        References:
            https://arxiv.org/abs/quant-ph/0406176 Theorem 12

        Args:
            mapping (Sequence[int]): Qubit ordering.

        Returns:
            Tuple[BasicGate]: Synthesized gates.
        """
        pass
