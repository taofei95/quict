#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 10:45 下午
# @Author  : Han Yu
# @File    : uniformRotation.py

from typing import *
import numpy as np

from .._synthesis import Synthesis
from QuICT.core import Qureg
from QuICT.core.gate import GateType, build_gate, CompositeGate


def uniformlyRotation(
        low: int,
        high: int,
        z: List[float],
        gate_type: int,
        mapping: List[int],
        is_left_cnot: bool = False
) -> CompositeGate:
    """
    synthesis uniformlyRotation gate, bits range [low, high)

    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<float>): the list of angle y
        gate_type(int): the gateType (Rz or Ry)
        mapping(list<int>): the qubit order of gate
        is_left_cnot(bool): is cnot left decomposition
    Returns:
        gateSet: the synthesis gate list
    """
    return inner_uniformly_rotation(low, high, z, gate_type, mapping, True, is_left_cnot)


def inner_uniformly_rotation(
        low: int,
        high: int,
        z: List[float],
        gate_type: int,
        mapping: List[int],
        is_first_level: bool,
        is_left_cnot: bool = False
) -> CompositeGate:
    if low + 1 == high:
        gateA = build_gate(gate_type, mapping[low], float(z[0]))
        gates = CompositeGate()
        gates.append(gateA)
        return gates
    length = len(z) // 2
    # GateBuilder.setGateType(GATE_ID["CX"])
    # GateBuilder.setTargs(mapping[high - 1])
    # GateBuilder.setCargs(mapping[low])
    q = Qureg([mapping[low], mapping[high - 1]])
    gateA = build_gate(GateType.cx, q)
    gateB = build_gate(GateType.cx, q)
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((z[i] + z[i + length]) / 2)
        Rxn.append((z[i] - z[i + length]) / 2)
    if is_first_level:
        if is_left_cnot:
            gates = CompositeGate()
            gates.append(gateA)
            gates.extend(inner_uniformly_rotation(low + 1, high, Rxn, gate_type, mapping, False, False))
            gates.append(gateB)
            gates.extend(inner_uniformly_rotation(low + 1, high, Rxp, gate_type, mapping, False, True))
        else:
            gates = inner_uniformly_rotation(low + 1, high, Rxp, gate_type, mapping, False, False)
            gates.append(gateA)
            gates.extend(inner_uniformly_rotation(low + 1, high, Rxn, gate_type, mapping, False, True))
            gates.append(gateB)
    elif is_left_cnot:
        gates = inner_uniformly_rotation(low + 1, high, Rxn, gate_type, mapping, False, False)
        gates.append(gateB)
        gates.extend(inner_uniformly_rotation(low + 1, high, Rxp, gate_type, mapping, False, True))
    else:
        gates = inner_uniformly_rotation(low + 1, high, Rxp, gate_type, mapping, False, False)
        gates.append(gateA)
        gates.extend(inner_uniformly_rotation(low + 1, high, Rxn, gate_type, mapping, False, True))
    return gates


class UniformlyRy(Synthesis):
    @classmethod
    def execute(cls, angle_list, mapping=None):
        """ uniformRyGate

        http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)

        Args:
            angle_list(list<float>): the angles of Ry Gates
            mapping(list<int>) : the mapping of gates order
        Returns:
            gateSet: the synthesis gate list
        """
        pargs = list(angle_list)
        n = int(np.round(np.log2(len(pargs)))) + 1
        if mapping is None:
            mapping = [i for i in range(n)]
        if 1 << (n - 1) != len(pargs):
            raise Exception("the number of parameters unmatched.")
        return uniformlyRotation(0, n, pargs, GateType.ry, mapping)


class UniformlyRz(Synthesis):
    @classmethod
    def execute(cls, angle_list, mapping=None):
        """ uniformRzGate

        http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 a)

        Args:
            angle_list(list<float>): the angles of Rz Gates
            mapping(list<int>) : the mapping of gates order
        Returns:
            gateSet: the synthesis gate list
        """
        pargs = list(angle_list)
        n = int(np.round(np.log2(len(pargs)))) + 1
        if mapping is None:
            mapping = [i for i in range(n)]
        if 1 << (n - 1) != len(pargs):
            raise Exception("the number of parameters unmatched.")
        return uniformlyRotation(0, n, pargs, GateType.rz, mapping)
