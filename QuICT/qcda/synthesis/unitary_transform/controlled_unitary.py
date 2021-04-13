from typing import *
import numpy as np

from .._synthesis import Synthesis
from ..uniformly_gate import uniformlyRz
from .utility import *

from QuICT.core import *


def __i_tensor_unitary(
        u: np.ndarray,
        recursive_basis: int,
        keep_left_diagonal: bool = False,
) -> Tuple[CompositeGate, complex]:
    """
    Transform (I_{2x2} tensor U) into gates. The 1st bit
    is under the identity transform.

    Args:
        u (np.ndarray): A unitary matrix.

    Returns:
        Tuple[CompositeGate, complex]: Synthesized gates and a phase factor.
    """

    gates: CompositeGate
    shift: complex
    # Dynamically import to avoid circulation.
    from .unitary_transform import inner_utrans_build_gate
    gates, shift = inner_utrans_build_gate(
        mat=u,
        recursive_basis=recursive_basis,
        keep_left_diagonal=keep_left_diagonal,
    )
    for gate in gates:
        for idx, _ in enumerate(gate.cargs):
            gate.cargs[idx] += 1
        for idx, _ in enumerate(gate.targs):
            gate.targs[idx] += 1

    return gates, shift


def inner_cutrans_build_gate(
        u1: np.ndarray,
        u2: np.ndarray,
        recursive_basis: int = 1,
        keep_left_diagonal: bool = False,
) -> Tuple[CompositeGate, complex]:
    """
    Build gates from parameterized model without mapping

    Returns:
        Tuple[CompositeGate, complex]: Synthesized gates and factor shift.
    """

    qubit_num = 1 + int(round(np.log2(u1.shape[0])))

    v, d, w = quantum_shannon_decompose(u1, u2)

    shift: complex = 1.0

    # diag(u1, u2) == diag(v, v) @ diag(d, d_dagger) @ diag(w, w)

    # diag(v, v)
    v_gates, _shift = __i_tensor_unitary(v, recursive_basis, keep_left_diagonal=True)
    shift *= _shift

    # diag(d, d_dagger)
    angle_list = []
    for i in range(d.shape[0]):
        s = d[i, i]
        theta = -2 * np.log(s) / 1j
        angle_list.append(theta)

    reversed_rz = uniformlyRz(
        angle_list=angle_list,
        mapping=[(i + 1) % qubit_num for i in range(qubit_num)]
    )

    # diag(w, w)
    if recursive_basis == 2:
        forwarded_d_gate: BasicGate = v_gates.pop(0)
        forwarded_mat = forwarded_d_gate.matrix
        for i in range(0, w.shape[0], 4):
            for k in range(4):
                w[i + k, :] *= forwarded_mat[k, k]

    w_gates, _shift = __i_tensor_unitary(w, recursive_basis, keep_left_diagonal=keep_left_diagonal)
    shift *= _shift

    gates = CompositeGate()
    gates.extend(w_gates)
    gates.extend(reversed_rz)
    gates.extend(v_gates)

    return gates, shift


def controlled_unitary_transform(
        u1: np.ndarray,
        u2: np.ndarray,
        recursive_basis: int = 1,
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
    qubit_num = 1 + int(round(np.log2(u1.shape[0])))
    gates, shift = inner_cutrans_build_gate(u1, u2, recursive_basis)
    if mapping is None:
        mapping = [i for i in range(qubit_num)]
    mapping = list(mapping)
    gates.remapping(mapping)
    if include_phase_gate:
        gates = add_factor_shift_into_phase(gates, shift)
        return gates
    else:
        return gates, shift


CUTrans = Synthesis(controlled_unitary_transform)
