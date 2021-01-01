#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/28 12:29 上午
# @Author  : Han Yu
# @File    : uniformly_unitary.py

import numpy as np

from .._synthesis import Synthesis
from QuICT.core import GateBuilder, GATE_ID
from QuICT.qcda.synthesis import uniformlyRz

def gates_from_unitary(unitary):
    """ gates from a one-qubit unitary

    Args:
        unitary(np.ndarray): the unitary to be transformed
    Returns:
        list<BasicGate>: gates from the unitary
    """
    return []

def get_parameters_from_unitaries(u1, u2):
    """ decomposition uniformly controlled one qubit unitaries

    Args:
        u1(np.ndarray): unitary with 0
        u2(np.ndarray): unitary with 1
    Returns:
        np.ndarray: v in the decomposition
        np.ndarray: u in the decomposition
        list<float>: angle list of Rz
    """
    return np.ndarray(), np.ndarray(), []

def uniformlyUnitary(low, high, unitary):
    """ synthesis uniformlyUnitary gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        unitary(list<int>): the list of unitaries
    Returns:
        the synthesis result
    """
    if low + 1 == high:
        return [gates_from_unitary(unitary[0])]
    length = len(z) // 2
    GateBuilder.setGateType(GATE_ID["CX"])
    GateBuilder.setTargs(high - 1)
    GateBuilder.setCargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    angle_list = []
    for i in range(length):
        u, v, angles = get_parameters_from_unitaries(unitary[i], unitary[i + length])
        angle_list.extend(angles)
        Rxp.append(u)
        Rxn.append(v)
    gates = uniformlyUnitary(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyUnitary(low + 1, high, Rxn))
    gates.append(gateB)
    gates.extend(uniformlyRz(angle_list).build_gate())
    return gates

class uniformlyUnitaryGate(Synthesis):
    """ uniformUnitaryGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 b)
    """

    def __call__(self, unitary_list):
        """
        Args:
            unitary_list(list<np.ndarray>): the angles of Unitary Gates
        Returns:
            uniformlyUnitaryGate: model filled by the parameter angle_list.
        """
        self.pargs = unitary_list
        return self

    def build_gate(self):
        """ overloaded the function "build_gate"

        """
        n = self.targets
        if 1 << (n - 1) != len(self.pargs):
            raise Exception("the number of parameters unmatched.")
        return uniformlyUnitary(0, n, self.pargs)
