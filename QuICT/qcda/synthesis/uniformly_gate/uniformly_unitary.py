#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/28 12:29 上午
# @Author  : Han Yu
# @File    : uniformly_unitary.py

import numpy as np

from . import UniformlyRz
from .._synthesis import Synthesis
from QuICT.core import Qureg
from QuICT.core.gate import build_gate, GateType, CompositeGate, H, Rz, U3


def gates_from_unitary(unitary, target):
    """ gates from a one-qubit unitary

    Args:
        unitary(np.ndarray): the unitary to be transformed
        target(int): the qubit gate acts on
    Returns:
        U3Gate: gate from the unitary
    """
    unitary = np.mat(unitary).reshape(2, 2)
    z = np.exp(1j * np.angle(unitary[0, 0]))
    unitary = unitary / z

    theta = np.arccos(unitary[0, 0])
    sint = np.sin(theta)
    if abs(sint) >= 1e-6:
        lamda = np.angle(unitary[0, 1] / -sint)
        phi = np.angle(unitary[1, 0] / sint)
    else:
        lamda = 0
        phi = np.angle(unitary[1, 1] / np.cos(theta))
    gate = U3.copy()
    gate.pargs = [theta * 2, phi, lamda]
    gate.targs = [target]
    assert not np.any(abs(gate.matrix.reshape(2, 2) - unitary) > 1e-6)
    return gate


def u2_expression(X):
    """ express U(2) with SU(2) and phase

    exp(i * phi / 2) SU(2) = U(2)

    Args:
        X(np.matrix): U2 matrix
    Returns:
        float: phase angle
    """
    if abs(abs(X[0, 0])) < 1e-10:
        absX = -X[1, 0] * X[0, 1]
        phase = np.angle(absX) / 2
        X[:] /= np.exp(1j * phase)
    else:
        absX = X[0, 0] * X[1, 1]
        phase = np.angle(absX) / 2
        X[:] /= np.exp(1j * phase)

    return 2 * phase


def get_parameters_from_unitaries(u1, u2):
    """ decomposition uniformly controlled one qubit unitaries

    (0><0) ⊗ u1 + (1><1) ⊗ u2

    Args:
        u1(np.ndarray): unitary with 0
        u2(np.ndarray): unitary with 1
    Returns:
        np.ndarray: v in the decomposition
        np.ndarray: u in the decomposition
        list<float>: angle list of Rz
    """
    a = np.mat(u1).reshape(2, 2)
    b = np.mat(u2).reshape(2, 2)

    X = a * b.H

    phi = u2_expression(X)

    x1 = np.angle(X[0, 0])

    r11_angle = 1j / 2 * (-np.pi / 2 - phi / 2 - x1)
    r22_angle = 1j / 2 * (np.pi / 2 - phi / 2 + x1)
    r = np.mat(np.diag([np.exp(r11_angle), np.exp(r22_angle)]))

    rXr = r * X * r * np.exp(1j * phi / 2)

    lamda, hU = np.linalg.eig(rXr)

    if abs(abs(lamda[0] - 1j)) >= 1e-10:
        hU[:, [0, 1]] = hU[:, [1, 0]]

    u = np.mat(hU).reshape(2, 2)

    d = np.mat(np.diag([np.exp(1j * np.pi / 4), np.exp(-1j * np.pi / 4)]))

    v = d.H * u.H * r * a

    assert not np.any(abs(u.H * u - np.diag([1, 1])) > 1e-7)
    assert not np.any(abs(u * d * d * u.H - rXr) > 1e-7)
    assert not np.any(abs(a - r.H * u * d * v) > 1e-7)
    assert not np.any(abs(b - r * u * d.H * v) > 1e-7)
    assert not np.any(abs(X * np.exp(1j * phi / 2) - r.H * u * d * d * u.H * r.H) > 1e-7)

    v = H.matrix.reshape(2, 2) * v
    v[:] *= np.exp(-1j * np.pi / 4)
    u = u * Rz(-np.pi / 2).matrix.reshape(2, 2) * H.matrix.reshape(2, 2)

    return v, u, [-np.pi - 1.0 * (x1 + phi / 2), 1.0 * (x1 - phi / 2)]


def uniformlyUnitarySolve(low, high, unitary, mapping):
    """ synthesis uniformlyUnitary gate, bits range [low, high)

    Args:
        low(int): the left range low
        high(int): the right range high
        unitary(list<int>): the list of unitaries
        mapping(list<int>): the qubit order of gate
    Returns:
        the synthesis result
    """
    if low + 1 == high:
        return CompositeGate(gates_from_unitary(unitary[0], low))
    length = len(unitary) // 2
    gateA = build_gate(GateType.cx, [mapping[low], mapping[high - 1]])
    Rxv = []
    Rxu = []
    angle_list = [0] * 2 * length
    for i in range(length):
        v, u, angles = get_parameters_from_unitaries(unitary[i], unitary[i + length])
        Rxu.append(u)
        Rxv.append(v)
        dual_position = 0
        for j in range(high - low - 2):
            if (1 << j) & i:
                dual_position += 1 << (high - low - 2 - j - 1)
        angle_list[dual_position] = angles[0]
        angle_list[dual_position + length] = angles[1]
    gates = uniformlyUnitarySolve(low + 1, high, Rxv, mapping)
    gates.append(gateA)
    gates.extend(uniformlyUnitarySolve(low + 1, high, Rxu, mapping))
    gates.extend(UniformlyRz.execute(angle_list, [mapping[i] for i in range(high - 1, low - 1, -1)]))
    return gates


class UniformlyUnitary(Synthesis):
    @classmethod
    def execute(cls, angle_list, mapping=None):
        """ uniformUnitaryGate

        http://cn.arxiv.org/abs/quant-ph/0504100v1 Fig4 b)

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
        return uniformlyUnitarySolve(0, n, pargs, mapping)
