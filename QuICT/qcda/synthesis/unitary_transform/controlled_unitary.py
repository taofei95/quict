import numpy as np

from typing import *
from .._synthesis import Synthesis
from ..uniformly_gate import uniformlyRz
from QuICT.core import *


def quantum_shannon_decompose(
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


def __i_tensor_unitary(
        u: np.ndarray,
        recursive_basis: int
) -> Tuple[CompositeGate, complex]:
    """
    Transform (I_{2x2} tensor U) into gates. The 1st bit
    is under the identity transform.

    Args:
        u (np.ndarray): A unitary matrix.

    Returns:
        Tuple[CompositeGate, complex]: Synthesized gates and a phase factor.
    """

    # Dynamically import for avoiding circular import.
    from .unitary_transform import UTrans
    gates: CompositeGate
    shift: complex
    gates, shift = UTrans(
        mat=u,
        recursive_basis=recursive_basis,
        mapping=None,
        include_phase_gate=False
    )
    for gate in gates:
        for idx, _ in enumerate(gate.cargs):
            gate.cargs[idx] += 1
        for idx, _ in enumerate(gate.targs):
            gate.targs[idx] += 1

    return gates, shift


def __build_gate(
        u1: np.ndarray,
        u2: np.ndarray,
        recursive_basis: int = 1
) -> Tuple[CompositeGate, complex]:
    """
    Build gates from parameterized model without mapping

    Returns:
        Tuple[CompositeGate, complex]: Synthesized gates and factor shift.
    """

    qubit_num = 1 + int(round(np.log2(u1.shape[0])))

    v, d, w = quantum_shannon_decompose(u1, u2)
    gates = CompositeGate()
    _gates: CompositeGate
    shift: complex = 1.0 + 0.0j

    # diag(u1, u2) == diag(v, v) @ diag(d, d_dagger) @ diag(w, w)

    # diag(w, w)
    _gates, _shift = __i_tensor_unitary(w, recursive_basis)
    shift *= _shift
    gates.extend(_gates)

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

    gates.extend(reversed_rz)

    # diag(v, v)
    _gates, _shift = __i_tensor_unitary(v, recursive_basis)
    shift *= _shift
    gates.extend(_gates)

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
    gates, shift = __build_gate(u1, u2, recursive_basis)
    if mapping is None:
        mapping = [i for i in range(qubit_num)]
    mapping = list(mapping)
    gates.remapping(mapping)
    if include_phase_gate:
        phase = np.log(shift) / 1j
        phase_gate = Phase.copy()
        phase_gate.pargs = [phase]
        phase_gate.targs = [0]
        gates.append(phase_gate)
        return gates
    else:
        return gates, shift


CUTrans = Synthesis(controlled_unitary_transform)
