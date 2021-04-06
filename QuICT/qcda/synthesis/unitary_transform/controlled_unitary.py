import numpy as np

from typing import *
from .._synthesis import Synthesis
from QuICT.core import BasicGate
from ..uniformly_gate import uniformlyRz
from .mapping_builder import MappingBuilder
from QuICT.core import *


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
            Tuple[np.ndarray,np.ndarray,np.ndarray]
        """
        s = u1 @ u2.conj().T

        eig_values, v = np.linalg.eig(s)
        v_dagger = v.conj().T
        d = np.sqrt(np.diag(eig_values))

        # u1 @ u2_dagger == v @ d_square @ v_dagger

        w = d @ v_dagger @ u2

        return v, d, w


class ControlledUnitary(MappingBuilder):
    """Decompose a 1-bit-controlled generic unitary with multiple targets.
    """

    def __call__(
            self,
            u1: np.ndarray,
            u2: np.ndarray,
            recursive_basis: int = 1
    ) -> "ControlledUnitary":
        """
        Build a parameterized model.


        Args:
            u1 (np.ndarray): Operation when control bit is |0>
            u2 (np.ndarray): Operation when control bit is |1>

        Returns:
            ControlledUnitary: Model filled with parameters.
        """
        self.pargs = [u1, u2, recursive_basis]
        self.targets = int(round(np.log2(u1.shape[0])))

        return self

    def __i_tensor_unitary(
            self,
            u: np.ndarray,
            recursive_basis: int
    ) -> Tuple[List[BasicGate], complex]:
        """
        Transform (I_{2x2} tensor U) into gates. The 1st bit
        is under the identity transform.

        Args:
            u (np.ndarray): A unitary matrix.

        Returns:
            Tuple[List[BasicGate], complex]: Synthesized gates and a phase factor.
        """

        # Dynamically import for avoiding circular import.
        from .unitary_transform import UnitaryTransform, UTrans
        gates: List[BasicGate]
        shift: complex
        gates, shift = UTrans(u, recursive_basis) \
            .build_gate(mapping=None, include_phase_gate=False)
        for gate in gates:
            for idx, _ in enumerate(gate.cargs):
                gate.cargs[idx] += 1
            for idx, _ in enumerate(gate.targs):
                gate.targs[idx] += 1

        return gates, shift

    def build_gate(
            self,
            mapping: Sequence[int] = None,
            include_phase_gate: bool = True
    ):
        """
                Build gates from parameterized model according to given mapping.

                References:
                    https://arxiv.org/abs/quant-ph/0406176 Theorem 12

                Args:
                    mapping (Sequence[int]): Qubit ordering.
                    include_phase_gate (bool): Whether to add phase gate.

                Returns:
                    1. If include_phase_gate==True, return List[BasicGate] in which
                    a phase gate is inserted to align phase gap.
                    2. If include_phase_gate==False, return Tuple[List[BasicGate], complex]
                    which means a gate sequence and corresponding phase factor f=exp(ia).

                """
        qubit_num = 1 + int(round(np.log2(self.pargs[0].shape[0])))
        gates, shift = self.__build_gate()
        self.remap(qubit_num, gates, mapping)
        if include_phase_gate:
            phase = np.log(shift) / 1j
            phase_gate = Phase.copy()
            phase_gate.pargs = [phase]
            phase_gate.targs = [0]
            gates.append(phase_gate)
            return gates
        else:
            return gates, shift

    def __build_gate(
            self
    ) -> Tuple[List[BasicGate], complex]:
        """
        Build gates from parameterized model without mapping

        Returns:
            Tuple[List[BasicGate], complex]: Synthesized gates and factor shift.
        """
        u1: np.ndarray = self.pargs[0]
        u2: np.ndarray = self.pargs[1]
        recursive_basis: int = self.pargs[2]
        self.pargs = []

        qubit_num = 1 + int(round(np.log2(u1.shape[0])))

        v, d, w = QuantumShannonDecompose.decompose(u1, u2)
        gates = []
        _gates = []
        shift: complex = 1.0 + 0.0j

        # diag(u1, u2) == diag(v, v) @ diag(d, d_dagger) @ diag(w, w)

        # diag(w, w)
        _gates, _shift = self.__i_tensor_unitary(w, recursive_basis)
        shift *= _shift
        gates.extend(_gates)

        # diag(d, d_dagger)
        angle_list = []
        for i in range(d.shape[0]):
            s = d[i, i]
            theta = -2 * np.log(s) / 1j
            angle_list.append(theta)

        reversed_rz = uniformlyRz(angle_list=angle_list) \
            .build_gate(mapping=[(i + 1) % qubit_num for i in range(qubit_num)])

        gates.extend(reversed_rz)

        # diag(v, v)
        _gates, _shift = self.__i_tensor_unitary(v, recursive_basis)
        shift *= _shift
        gates.extend(_gates)

        return gates, shift


CUTrans = ControlledUnitary()
