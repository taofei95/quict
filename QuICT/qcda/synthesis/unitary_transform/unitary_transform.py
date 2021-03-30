"""
Decomposition of unitary matrix U∈SU(2^n)
"""

import numpy as np

# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from typing import *
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary
from .two_qubit_transform import KAK
from ..uniformly_gate import uniformlyRy
from .mapping_builder import MappingBuilder


class UnitaryTransform(MappingBuilder):
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

    def __init__(self):
        super().__init__()

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

    def build_gate(
            self,
            mapping: Sequence[int] = None,
            include_phase_gate: bool = True
    ):
        """
        Return:
            1. If include_phase_gate==True, return List[BasicGate] in which
            a phase gate is inserted to align phase gap.
            2. If include_phase_gate==False, return Tuple[List[BasicGate], complex]
            which means a gate sequence and corresponding phase factor f=exp(ia).
        """
        qubit_num = int(round(np.log2(self.pargs[0].shape[0])))
        basis = self.pargs[1]
        gates, shift = self.__build_gate()
        self.remap(qubit_num, gates, mapping)
        gates = list(gates)

        if basis == 2 and include_phase_gate:
            phase = np.log(shift) / 1j
            phase_gate = Phase.copy()
            phase_gate.pargs = [phase]
            phase_gate.targs = [0]
            gates.append(phase_gate)
        if include_phase_gate:
            return gates
        else:
            return gates, shift

    def __build_gate(
            self
    ) -> Tuple[List[BasicGate], complex]:
        """
        No mapping
        """

        mat: np.ndarray = np.array(self.pargs[0])
        recursive_basis: int = self.pargs[1]
        eps: float = self.pargs[2]
        # clear pargs
        self.pargs = []
        mat_size: int = mat.shape[0]
        qubit_num = int(round(np.log2(mat_size)))

        if qubit_num == 1:
            GateBuilder.setGateType(GATE_ID["Unitary"])
            # TODO: Unitary Gate matrix type restrictions
            parg = np.reshape(mat, -1).tolist()
            GateBuilder.setPargs(parg)
            GateBuilder.setTargs([0])
            u = GateBuilder.getGate()
            return [u], 1.0 + 0.0j
        elif qubit_num == 2 and recursive_basis == 2:
            gates: List[BasicGate] = KAK(mat).build_gate()
            # TODO: Avoid build a circuit.
            circuit = Circuit(qubit_num)
            circuit.extend(gates)
            syn_mat = SyntheticalUnitary.run(circuit)
            shift: complex = mat[0, 0] / syn_mat[0, 0]
            # self.factor_shift *= shift
            del circuit, syn_mat
            return gates, shift

        gates = []

        u, angle_list, v_dagger = cossin(mat, mat_size // 2, mat_size // 2, separate=True)

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
        v1_dagger = v_dagger[0]
        v2_dagger = v_dagger[1]

        _gates = []
        shift: complex = 1.0

        # Dynamically import for avoiding circular import.
        from .controlled_unitary import ControlledUnitary, CUTrans
        _gates, _shift = CUTrans(v1_dagger, v2_dagger, recursive_basis) \
            .build_gate(include_phase_gate=False)
        shift *= _shift
        gates.extend(_gates)

        # (c,s\\s,c)
        angle_list *= 2  # Ry use its angle as theta/2
        reversed_ry = uniformlyRy(angle_list=angle_list) \
            .build_gate(mapping=[(i + 1) % qubit_num for i in range(qubit_num)])
        gates.extend(reversed_ry)

        # u
        u1 = u[0]
        u2 = u[1]

        _gates, _shift = CUTrans(u1, u2, recursive_basis) \
            .build_gate(include_phase_gate=False)
        shift *= _shift
        gates.extend(_gates)

        return gates, shift


UTrans = UnitaryTransform()
