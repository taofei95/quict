#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 10:45 下午
# @Author  : Han Yu
# @File    : uniformRotation.py

import numpy as np

from .._synthesis import Synthesis
from QuICT.core import GATE_ID, GateBuilder, GateSet

def uniformlyRotation(low, high, z, gateType, mapping):
    """ synthesis uniformlyRotation gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<int>): the list of angle y
        gateType(int): the gateType (Rz or Ry)
        mapping(list<int>): the qubit order of gate
    Returns:
        gateSet: the synthesis gate list
    """
    if low + 1 == high:
        GateBuilder.setGateType(gateType)
        GateBuilder.setTargs(mapping[low])
        GateBuilder.setPargs(float(z[0]))
        return GateSet(GateBuilder.getGate())
    length = len(z) // 2
    GateBuilder.setGateType(GATE_ID["CX"])
    GateBuilder.setTargs(mapping[high - 1])
    GateBuilder.setCargs(mapping[low])
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((z[i] + z[i + length]) / 2)
        Rxn.append((z[i] - z[i + length]) / 2)
    gates = uniformlyRotation(low + 1, high, Rxp, gateType, mapping)
    gates.append(gateA)
    gates.extend(uniformlyRotation(low + 1, high, Rxn, gateType, mapping))
    gates.append(gateB)
    return gates

def uniformlyRyDecomposition(angle_list, n, mapping = None):
    """ uniformRyGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)

    Args:
        angle_list(list<float>): the angles of Ry Gates
        n(int) : the number of targets
        mapping(list<int>) : the mapping of gates order
    Returns:
        gateSet: the synthesis gate list
    """
    pargs = list(angle_list)
    if mapping is None:
        mapping = [i for i in range(n)]
    if 1 << (n - 1) != len(pargs):
        raise Exception("the number of parameters unmatched.")
    return uniformlyRotation(0, n, pargs, GATE_ID['Ry'], mapping)

uniformlyRy = Synthesis(uniformlyRyDecomposition)

def uniformlyRzDecomposition(angle_list, n, mapping = None):
    """ uniformRzGate

    http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)

    Args:
        angle_list(list<float>): the angles of Rz Gates
        n(int) : the number of targets
        mapping(list<int>) : the mapping of gates order
    Returns:
        gateSet: the synthesis gate list
    """
    pargs = list(angle_list)
    if mapping is None:
        mapping = [i for i in range(n)]
    if 1 << (n - 1) != len(pargs):
        raise Exception("the number of parameters unmatched.")
    return uniformlyRotation(0, n, pargs, GATE_ID['Rz'], mapping)

uniformlyRz = Synthesis(uniformlyRzDecomposition)
