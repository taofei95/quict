"""
Decomposition of uniformly-Ry gate, which is specially designed for 
the optimization of unitary transform
"""

from typing import *
import numpy as np

from QuICT.core import CompositeGate, CX, CZ, Ry, Rz
from .._synthesis import Synthesis


def uniformly_rotation_cz(
        low: int,
        high: int,
        z: List[float],
        mapping: List[int],
        is_cz_left: bool = False
) -> CompositeGate:
    """
    synthesis uniformlyRy gate, bits range [low, high)

    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<float>): the list of angle y
        mapping(list<int>): the qubit order of gate
        is_cz_left(bool): is cx/cz left decomposition
    Returns:
        gateSet: the synthesis gate list
    """
    return inner_uniformly_rotation_cz(low, high, z, mapping, True, is_cz_left)


def inner_uniformly_rotation_cz(
        low: int,
        high: int,
        z: List[float],
        mapping: List[int],
        is_first_level: bool,
        is_cz_left: bool = False
) -> CompositeGate:
    if low + 1 == high:
        gates = CompositeGate()
        with gates:
            Ry(float(z[0])) & mapping[low]
        return gates
    length = len(z) // 2
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((z[i] + z[i + length]) / 2)
        Rxn.append((z[i] - z[i + length]) / 2)
    if is_first_level:
        if is_cz_left:
            gates = CompositeGate()

            with gates:
                CZ & [mapping[low], mapping[high - 1]]
                # if high - low == 2, no CZ is needed here
                if high - low > 2:
                    CZ & [mapping[low], mapping[low + 1]]
            gates.extend(inner_uniformly_rotation_cz(low + 1, high, Rxn, mapping, False, False))

            with gates:
                # CZ & [mapping[low], mapping[high - 1]]
                Ry(np.pi / 2) & mapping[high - 1]
                CX & [mapping[low], mapping[high - 1]]
                Ry(-np.pi / 2) & mapping[high - 1]
                # The extra CZ derived from CCPhase can also be moved to the edge

            gates.extend(inner_uniformly_rotation_cz(low + 1, high, Rxp, mapping, False, True))
        else:
            gates = inner_uniformly_rotation_cz(low + 1, high, Rxp, mapping, False, False)
            with gates:
                # CZ & [mapping[low], mapping[high - 1]]
                Ry(np.pi / 2) & mapping[high - 1]
                CX & [mapping[low], mapping[high - 1]]
                Ry(-np.pi / 2) & mapping[high - 1]
                # The extra CZ derived from CCPhase can also be moved to the edge
            gates.extend(inner_uniformly_rotation_cz(low + 1, high, Rxn, mapping, False, True))

            with gates:
                CZ & [mapping[low], mapping[high - 1]]
                # if high - low == 2, no CZ is needed here
                if high - low > 2:
                    CZ & [mapping[low], mapping[low + 1]]
    elif is_cz_left:
        gates = inner_uniformly_rotation_cz(low + 1, high, Rxn, mapping, False, False)
        with gates:
            CX & [mapping[low], mapping[high - 1]]
        gates.extend(inner_uniformly_rotation_cz(low + 1, high, Rxp, mapping, False, True))
    else:
        gates = inner_uniformly_rotation_cz(low + 1, high, Rxp, mapping, False, False)
        with gates:
            CX & [mapping[low], mapping[high - 1]]
        gates.extend(inner_uniformly_rotation_cz(low + 1, high, Rxn, mapping, False, True))
    return gates


def uniformlyRyDecompostionRevision(angle_list, mapping=None, is_cz_left: bool = False):
    """ uniformlyRyGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)
    This part is mainly copied from ../uniformly_gate/uniformly_rotation.py
    Here we demand a CZ gate at the edge of the decomposition, therefore the
    recursion process is slightly revised.

    Args:
        angle_list(list<float>): the angles of Ry Gates
        mapping(list<int>) : the mapping of gates order
    Returns:
        CompositeGate: the synthesis gate list
    """
    pargs = list(angle_list)
    n = int(np.round(np.log2(len(pargs)))) + 1
    if mapping is None:
        mapping = [i for i in range(n)]
    if 1 << (n - 1) != len(pargs):
        raise Exception("the number of parameters unmatched.")
    return uniformly_rotation_cz(0, n, pargs, mapping, is_cz_left)


uniformlyRyRevision = Synthesis(uniformlyRyDecompostionRevision)
"""
If qubit_num > 2, synthesized gates would have 2 cz gates at rightmost place.
If qubit_num == 2, there would be only 1 cz gate.
"""
