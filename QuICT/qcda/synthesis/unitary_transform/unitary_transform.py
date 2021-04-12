"""
Decomposition of unitary matrix U∈SU(2^n)
"""

import numpy as np

# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from typing import *
from QuICT.core import *
from .two_qubit_transform import KAK
from ..uniformly_gate import uniformlyRy
from .._synthesis import Synthesis
from .uniformly_ry_revision import uniformlyRyDecompostionRevision


def __build_gate(
        mat: np.ndarray,
        recursive_basis: int = 1
) -> Tuple[CompositeGate, complex]:
    """
    No mapping
    """

    mat: np.ndarray = np.array(mat)
    mat_size: int = mat.shape[0]
    qubit_num = int(round(np.log2(mat_size)))

    if qubit_num == 1:
        GateBuilder.setGateType(GATE_ID["Unitary"])
        parg = np.reshape(mat, -1).tolist()
        GateBuilder.setPargs(parg)
        GateBuilder.setTargs([0])
        u = GateBuilder.getGate()
        _ret = CompositeGate(gates=[u])
        return _ret, 1.0 + 0.0j
    elif qubit_num == 2 and recursive_basis == 2:
        gates = KAK(mat)
        syn_mat = gates.matrix()
        shift = 1.0 + 0.0j
        for j in range(mat_size):
            if not np.isclose(0, syn_mat[0, j]):
                shift = mat[0, j] / syn_mat[0, j]
        return gates, shift

    gates = CompositeGate()

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

    _gates: CompositeGate
    shift: complex = 1.0

    # Dynamically import for avoiding circular import.
    from .controlled_unitary import controlled_unitary_transform, CUTrans
    _gates, _shift = controlled_unitary_transform(
        u1=v1_dagger,
        u2=v2_dagger,
        recursive_basis=recursive_basis,
        mapping=None,
        include_phase_gate=False
    )
    shift *= _shift
    gates.extend(_gates)

    # (c,s\\s,c)
    angle_list *= 2  # Ry use its angle as theta/2
    reversed_ry = uniformlyRyDecompostionRevision(
        angle_list=angle_list,
        mapping=[(i + 1) % qubit_num for i in range(qubit_num)],
        is_cz_left=False  # keep CZ at right side
    )
    gates.extend(reversed_ry)

    """
    Now, gates have CZ gate(s) at it's ending part.
    If qubit_num > 2, we would have gates[-1] as a CZ affecting on (0, 1), 
    while gates[-2] a CZ on (0, qubit_num - 1).
    If qubit_num == 2, there would only be one CZ affecting on (0, 1).
    """

    # u
    u1: np.ndarray = u[0]
    u2: np.ndarray = u[1]

    gates.pop()  # CZ on (0,1)
    # This CZ affects 1/4 last columns of the matrix of U, or 1/2 last columns of u2.
    _u_size = u2.shape[0]
    for j in range(_u_size // 2, _u_size):
        u2[:, j] = -u2[:, j]

    if qubit_num > 2:
        gates.pop()  # CZ on (0, qubit_num - 1)
        # For similar reasons, this CZ only affect 2 parts of matrix of U.
        for j in range(_u_size - _u_size // 4, _u_size):
            u1[:, j] = - u1[:, j]
            u2[:, j] = - u2[:, j]

    _gates, _shift = controlled_unitary_transform(
        u1=u1,
        u2=u2,
        recursive_basis=recursive_basis,
        mapping=None,
        include_phase_gate=False
    )
    shift *= _shift
    gates.extend(_gates)

    return gates, shift


def unitary_transform(
        mat: np.ndarray,
        recursive_basis: int = 1,
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
    qubit_num = int(round(np.log2(mat.shape[0])))
    basis = recursive_basis
    gates, shift = __build_gate(mat, recursive_basis)
    if mapping is None:
        mapping = [i for i in range(qubit_num)]
    mapping = list(mapping)
    gates.remapping(mapping)
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


UTrans = Synthesis(unitary_transform)
