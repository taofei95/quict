#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/28 12:29 上午
# @Author  : Han Yu
# @File    : uniformly_unitary.py

import numpy as np

from . import uniformlyRz
from .._synthesis import Synthesis
from QuICT.core import GateBuilder, GATE_ID, H, Rz, U3

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
    if abs(z) >= 1e-10:
        unitary[:] /= z
    theta = np.arccos(unitary[0, 0])
    sint = np.sin(theta)
    if abs(sint) >= 1e-10:
        lamda = np.angle(unitary[0, 1] / -sint)
        phi = np.angle(unitary[1, 0] / sint)
    else:
        lamda = 0
        phi = np.angle(unitary[1, 1] / np.cos(theta))
    gate = U3.copy()
    gate.pargs = [theta * 2, phi, lamda]
    gate.targs = [target]
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
        absX = X[1, 0] * X[0, 1]
        phase = 1 / absX
        X[:] /= np.sqrt(phase)
    else:
        phase = X[1, 1].conj() / X[0, 0]
        X[:] /= np.sqrt(phase)

    return np.angle(phase) * 2

def get_parameters_from_unitaries(u1, u2):
    """ decomposition uniformly controlled one qubit unitaries

    |0><0| ⊗ u1 + |1><1| ⊗ u2

    Args:
        u1(np.ndarray): unitary with 0
        u2(np.ndarray): unitary with 1
    Returns:
        np.ndarray: v in the decomposition
        np.ndarray: u in the decomposition
        list<float>: angle list of Rz
    """
    print()
    a = np.mat(u1).reshape(2, 2)
    b = np.mat(u2).reshape(2, 2)
    X = a * b.H
    phi = u2_expression(X)

    x1 = np.angle(X[0, 0])
    r11_angle = 1j / 2 * (-np.pi / 2 - phi / 2 - x1)
    r22_angle = 1j / 2 * (np.pi / 2 - phi / 2 + x1)
    r = np.mat(np.diag([np.exp(r11_angle), np.exp(r22_angle)]))
    rXr = r * X * r
    lamda, hU = np.linalg.eig(rXr)
    if abs(abs(lamda[0] - 1j)) >= 1e-10:
        hU[[0, 1], :] = hU[[1, 0], :]
    u = np.mat(hU).reshape(2, 2)

    d = np.mat(np.diag([np.exp(1j * np.pi / 4), np.exp(-1j * np.pi / 4)]))
    v = d.H * u.H * r * a

    v = H.matrix.reshape(2, 2) * v
    v[:] *= np.exp(-1j * np.pi / 4)

    u = u * Rz(np.pi / 2).matrix.reshape(2, 2) * H.matrix.reshape(2, 2)

    return v, u, [-1.0 / 2 * (x1 + phi / 2), np.pi / 2 + 1.0 / 2 * (x1 - phi / 2)]

def uniformlyUnitarySolve(low, high, unitary):
    """ synthesis uniformlyUnitary gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        unitary(list<int>): the list of unitaries
    Returns:
        the synthesis result
    """
    if low + 1 == high:
        return [gates_from_unitary(unitary[0], low)]
    length = len(unitary) // 2
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
    gates = uniformlyUnitarySolve(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyUnitarySolve(low + 1, high, Rxn))
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
        return uniformlyUnitarySolve(0, n, self.pargs)

uniformlyUnitary = uniformlyUnitaryGate()
