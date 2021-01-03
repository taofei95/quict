#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 10:45 下午
# @Author  : Han Yu
# @File    : uniformRotation.py

import numpy as np

from .._synthesis import Synthesis
from QuICT.core import GateBuilder, GATE_ID

def uniformlyRotation(low, high, z, gateType):
    """ synthesis uniformlyRotation gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<int>): the list of angle y
        gateType(int): the gateType (Rz or Ry)
    Returns:
        the synthesis result
    """
    if low + 1 == high:
        GateBuilder.setGateType(gateType)
        GateBuilder.setTargs(low)
        GateBuilder.setPargs(z[0])
        return [GateBuilder.getGate()]
    length = len(z) // 2
    GateBuilder.setGateType(GATE_ID["CX"])
    GateBuilder.setTargs(high - 1)
    GateBuilder.setCargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((z[i] + z[i + length]) / 2)
        Rxn.append((z[i] - z[i + length]) / 2)
    gates = uniformlyRotation(low + 1, high, Rxp, gateType)
    gates.append(gateA)
    gates.extend(uniformlyRotation(low + 1, high, Rxn, gateType))
    gates.append(gateB)
    return gates

class uniformlyRyGate(Synthesis):
    """ uniformRyGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)
    """

    def __call__(self, angle_list):
        """
        Args:
            angle_list(list<float>): the angles of Ry Gates
        Returns:
            uniformlyRyGate: model filled by the parameter angle_list.
        """
        self.pargs = angle_list
        self.targets = int(np.round(np.log2(len(self.pargs)))) + 1
        return self

    def build_gate(self):
        """ overloaded the function "build_gate"

        """
        n = self.targets
        if 1 << (n - 1) != len(self.pargs):
            raise Exception("the number of parameters unmatched.")
        return uniformlyRotation(0, n, self.pargs, GATE_ID['Ry'])

uniformlyRy = uniformlyRyGate()

class uniformlyRzGate(Synthesis):
    """ uniformRzGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)
    """

    def __call__(self, angle_list):
        """
        Args:
            angle_list(list<float>): the angles of Rz Gates
        Returns:
            uniformlyRzGate: model filled by the parameter angle_list.
        """
        self.pargs = angle_list
        self.targets = int(np.round(np.log2(len(self.pargs)))) + 1
        return self

    def build_gate(self):
        """ overloaded the function "build_gate"

        """
        n = self.targets
        if 1 << (n - 1) != len(self.pargs):
            raise Exception("the number of parameters unmatched.")
        return uniformlyRotation(0, n, self.pargs, GATE_ID['Rz'])

uniformlyRz = uniformlyRzGate()
