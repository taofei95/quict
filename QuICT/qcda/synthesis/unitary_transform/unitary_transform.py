from typing import *
import numpy as np

from scipy.linalg import cossin
from QuICT.core import *

from .two_qubit_transform import KAK
from .two_qubit_diagonal_transform import KAKDiag
from ..uniformly_gate import uniformlyRy
from .._synthesis import Synthesis
from .uniformly_ry_revision import uniformlyRyDecompostionRevision
from .utility import *


def inner_utrans_build_gate(
        mat: np.ndarray,
        recursive_basis: int = 1,
        keep_left_diagonal: bool = False,
        use_cz_ry: bool = True,
) -> Tuple[CompositeGate, complex]:
    """
    No mapping
    """

    mat: np.ndarray = np.array(mat)
    mat_size: int = mat.shape[0]
    qubit_num = int(round(np.log2(mat_size)))

    _kak = KAKDiag if keep_left_diagonal else KAK

    if qubit_num == 1:
        GateBuilder.setGateType(GATE_ID["Unitary"])
        parg = np.reshape(mat, -1).tolist()
        GateBuilder.setPargs(parg)
        GateBuilder.setTargs([0])
        u = GateBuilder.getGate()
        _ret = CompositeGate(gates=[u])
        return _ret, 1.0 + 0.0j
    elif qubit_num == 2 and recursive_basis == 2:
        gates = _kak(mat)
        syn_mat = gates.matrix()
        shift = shift_ratio(mat, syn_mat)
        return gates, shift

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

    # Dynamically import to avoid circulation.
    from .controlled_unitary import inner_cutrans_build_gate

    shift: complex = 1.0

    # (c,s\\s,c)
    angle_list *= 2  # Ry use its angle as theta/2
    if use_cz_ry:
        reversed_ry = uniformlyRyDecompostionRevision(
            angle_list=angle_list,
            mapping=[(i + 1) % qubit_num for i in range(qubit_num)],
            is_cz_left=False,  # keep CZ at right side
        )
    else:
        reversed_ry = uniformlyRy(
            angle_list=angle_list,
            mapping=[(i + 1) % qubit_num for i in range(qubit_num)],
        )

    """
    Now, gates have CZ gate(s) at it's ending part(the left side of u).
    Left gate of u is right multiplied to u in matrix view.
    If qubit_num > 2, we would have reversed_ry[-1] as a CZ affecting on (0, 1), 
    while reversed_ry[-2] a CZ on (0, qubit_num - 1).
    If qubit_num == 2, there would only be one CZ affecting on (0, 1).
    """

    # u
    u1: np.ndarray = u[0]
    u2: np.ndarray = u[1]

    if use_cz_ry:
        reversed_ry.pop()  # CZ on (0,1)
        # This CZ affects 1/4 last columns of the matrix of U, or 1/2 last columns of u2.
        _u_size = u2.shape[0]
        for i in range(_u_size // 2, _u_size):
            u2[:, i] = -u2[:, i]

        if qubit_num > 2:
            reversed_ry.pop()  # CZ on (0, qubit_num - 1)
            # For similar reasons, this CZ only affect 2 parts of matrix of U.
            for i in range(_u_size - _u_size // 4, _u_size):
                u1[:, i] = - u1[:, i]
                u2[:, i] = - u2[:, i]

    u_gates, _shift = inner_cutrans_build_gate(
        u1=u1,
        u2=u2,
        recursive_basis=recursive_basis,
        keep_left_diagonal=True,
    )
    shift *= _shift

    # v_dagger

    """
    Now, leftmost gate decompose by u is a diagonal gate on (n-2, n-1), which 
    is commutable with reversed_ry. That gate is in right side of v_dagger, so 
    it would be left multiplied to v_dagger.
    """
    v1_dagger = v_dagger[0]
    v2_dagger = v_dagger[1]

    if recursive_basis == 2:
        forwarded_d_gate: BasicGate = u_gates.pop(0)
        forwarded_mat = forwarded_d_gate.matrix
        for i in range(0, mat_size // 2, 4):
            for k in range(4):
                v1_dagger[i + k, :] *= forwarded_mat[k, k]
                v2_dagger[i + k, :] *= forwarded_mat[k, k]

    v_dagger_gates, _shift = inner_cutrans_build_gate(
        u1=v1_dagger,
        u2=v2_dagger,
        recursive_basis=recursive_basis,
        keep_left_diagonal=keep_left_diagonal,
    )
    shift *= _shift

    gates = CompositeGate()
    gates.extend(v_dagger_gates)
    gates.extend(reversed_ry)
    gates.extend(u_gates)

    return gates, shift


def unitary_transform(
        mat: np.ndarray,
        include_phase_gate: bool = True,
        mapping: List[int] = None,
        recursive_basis: int = 2,
):
    """
    Transform a general unitary matrix into CX gates and single qubit gates.

    Args:
        mat(np.ndarray): Unitary matrix.
        include_phase_gate(bool): Whether to include a phase gate to keep synthesized gate matrix the same
            as input. If set False, the output gates might have a matrix which has a factor shift to input:
            np.allclose(<matrix_of_return_gates> * factor, <input_matrix>).
        mapping(List[int]): The order of input qubits. Mapping is a list of their labels from top to bottom.
        recursive_basis(int): Terminate recursion at which level. It could be set as 1 or 2, which would stop
            recursion when matrix is 2 or 4, respectively. When set as 2, the final step is done by KAK decomposition.
            Correctness of this algorithm is never influenced by recursive_basis.

    Returns:
        Union[CompositeGate, Tuple[CompositeGate,complex]]: If inlclude_phase_gate==True, this function returns
            synthesized gates. Otherwise gates and a factor would be returned.
    """
    qubit_num = int(round(np.log2(mat.shape[0])))

    """
    After adding KAK diagonal optimization, inner built gates would have a 
    leading diagonal gate not decomposed. If our first-level recursion is 
    2-bit, using older version of KAK is OK.
    """

    if qubit_num == 2 and recursive_basis == 2:
        gates = KAK(mat)
        syn_mat = gates.matrix()
        shift = shift_ratio(mat, syn_mat)
        add_factor_shift_into_phase(gates, shift)
        return gates

    gates, shift = inner_utrans_build_gate(
        mat=mat,
        recursive_basis=recursive_basis,
        keep_left_diagonal=False,
    )
    if mapping is None:
        mapping = [i for i in range(qubit_num)]
    mapping = list(mapping)

    gates.remapping(mapping)
    if recursive_basis == 2 and include_phase_gate:
        gates = add_factor_shift_into_phase(gates, shift)
    if include_phase_gate:
        return gates
    else:
        return gates, shift


UTrans = Synthesis(unitary_transform)
