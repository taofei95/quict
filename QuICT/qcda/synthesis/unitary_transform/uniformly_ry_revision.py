"""
Decomposition of uniformly-Ry gate, which is specially designed for 
the optimization of unitary transform
"""

import numpy as np

from QuICT.core import CompositeGate, CX, CZ, Ry, Rz
from .._synthesis import Synthesis


def uniformlyRotation(low, high, z, mapping, direction=None):
    """
    synthesis uniformlyRy gate, bits range [low, high)

    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<int>): the list of angle y
        mapping(list<int>): the qubit order of gate
        direction(bool): is cnot left decomposition
    Returns:
        gateSet: the synthesis gate list
    """
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
    if direction is None:
        gates = uniformlyRotation(low + 1, high, Rxp, mapping, False)
        with gates:
            CZ & [mapping[low], mapping[high - 1]]
            """ The extra CZ derived from CCPhase can also be moved to the edge
            # if high - low == 2, no CZ is needed here
            if high - low > 2:
                CZ & [mapping[low], mapping[low + 1]]
            """
        gates.extend(uniformlyRotation(low + 1, high, Rxn, mapping, True))

        with gates:
            CZ & [mapping[low], mapping[high - 1]]
            # if high - low == 2, no CZ is needed here
            if high - low > 2:
                CZ & [mapping[low], mapping[low + 1]]
    elif direction:
        gates = uniformlyRotation(low + 1, high, Rxn, mapping, False)
        with gates:
            CX & [mapping[low], mapping[high - 1]]
        gates.extend(uniformlyRotation(low + 1, high, Rxp, mapping, True))
    else:
        gates = uniformlyRotation(low + 1, high, Rxp, mapping, False)
        with gates:
            CX & [mapping[low], mapping[high - 1]]
        gates.extend(uniformlyRotation(low + 1, high, Rxn, mapping, True))
    return gates


def uniformlyRyDecompostionRevision(angle_list, mapping=None):
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
    return uniformlyRotation(0, n, pargs, mapping)


uniformlyRyRevision = Synthesis(uniformlyRyDecompostionRevision)
