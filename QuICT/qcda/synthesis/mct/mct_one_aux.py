#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 11:37
# @Author  : Han Yu
# @File    : MCT_one_aux.py

from .._synthesis import Synthesis
from QuICT.core import *
from QuICT.core.gate import *
from .mct_linear_simulation import MCTLinearHalfDirtyAux


def merge_qubit(qubit_a, qubit_b):
    """ merge two qureg into one in order

    Args:
        qubit_a(Qubit/Qureg): the first part of new qureg
        qubit_b(Qubit/Qureg): the second part of new qureg

    Returns:
        Qureg: the qureg merged by qubit_a and qubit_b
    """
    qureg = []
    if isinstance(qubit_a, Qubit):
        qureg.append(qubit_a)
    else:
        for qubit in qubit_a:
            qureg.append(qubit)
    if isinstance(qubit_b, Qubit):
        qureg.append(qubit_b)
    else:
        for qubit in qubit_b:
            qureg.append(qubit)
    return qureg


def solve(n):
    """ Decomposition of n-qubit Toffoli gates with one ancillary qubit and linear circuit complexity

    Args:
        n(int): the bit of toffoli gate
    Returns:
        the circuit which describe the decomposition result
    """
    qubit_list = Circuit(n + 1)
    if n == 3:
        CCX | qubit_list[:3]
        return qubit_list
    elif n == 2:
        CX | qubit_list[:2]
        return qubit_list
    if n % 2 == 1:
        k1 = n // 2 + 1
    else:
        k1 = n // 2
    k2 = n // 2 - 1

    MCTLinearHalfDirtyAux.execute(k1, n + 1) | qubit_list
    H | qubit_list[-2]
    S | qubit_list[-1]
    MCTLinearHalfDirtyAux.execute(k2 + 1, n + 1) | merge_qubit(
        merge_qubit(qubit_list[k1:k1 + k2 + 1], qubit_list[:k1]),
        qubit_list[-1]
    )
    S_dagger | qubit_list[-1]
    MCTLinearHalfDirtyAux.execute(k1, n + 1) | qubit_list
    S | qubit_list[-1]
    MCTLinearHalfDirtyAux.execute(k2 + 1, n + 1) | merge_qubit(
        merge_qubit(qubit_list[k1:k1 + k2 + 1], qubit_list[:k1]),
        qubit_list[-1]
    )
    H | qubit_list[-2]
    S_dagger | qubit_list[-1]

    return qubit_list


class MCTOneAux(Synthesis):
    @staticmethod
    def execute(n):
        """ Decomposition of n-qubit Toffoli gates with one ancillary qubit and linear circuit complexity

        He Y, Luo M X, Zhang E, et al.
        Decompositions of n-qubit Toffoli gates with linear circuit complexity[J].
        International Journal of Theoretical Physics, 2017, 56(7): 2350-2361.
        Args:
            n(int): the bits of the toffoli gate
        Return:
            CompositeGate: the result of Decomposition
        """
        gates = CompositeGate()
        gates.extend(solve(n - 1).gates)
        return gates
