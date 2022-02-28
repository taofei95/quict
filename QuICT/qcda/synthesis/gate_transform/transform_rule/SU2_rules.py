#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:42 下午
# @Author  : Han Yu
# @File    : SU2_rules.py

"""

the file describe TransformRule the decomposite SU(2) into instruction set.

"""

import random
import numpy as np
from numpy import arccos, linalg
from scipy.stats import ortho_group

from QuICT.core.gate import *

from .transform_rule import TransformRule


def _arccos(value):
    """ calculate arccos(value)

    Args:
        value: the cos value

    Returns:
        float: the corresponding angle
    """
    if value < -1:
        value = -1
    elif value > 1:
        value = 1
    return arccos(value)


def _check2pi(theta, eps=1e-15):
    """ check whether theta is a multiple of 2π

    Args:
        theta(float): the angle to be checked
        eps(float): tolerate error

    Returns:
        bool: whether theta is a multiple of 2π
    """
    multiple = np.round(theta / (2 * np.pi))
    return abs(2 * np.pi * multiple - theta) < eps


class SU2TransformRule(TransformRule):
    """ a subClass of TransformRule, to check the decomposition of SU2

    """

    def check_equal(self, ignore_phase=True, eps=1e-7):
        qubit = 1
        d = np.power(2, qubit)
        Q = np.mat(ortho_group.rvs(dim=d))
        Q = Q.reshape(d, d)
        diag1 = []
        diag2 = []
        for i in range(d):
            ran = random.random()
            diag1.append(ran)
            diag2.append(np.sqrt(1 - ran * ran))

        d1 = np.diag(diag1)
        d2 = np.diag(diag2)
        A = Q.T * d1 * Q
        B = Q.T * d2 * Q
        U = A + B[:] * 1j

        gate = Unitary(U)
        gate.targs = [0]
        compositeGate = self.transform(gate)
        ans = compositeGate.equal(gate, ignore_phase=ignore_phase, eps=eps)
        # if not ans:
        #     print()
        #     print("U:")
        #     print(U)
        #     print("Decomposed as:")
        #     prod = np.eye(2)
        #     for gate in compositeGate.gates:
        #         prod = gate.matrix @ prod
        #         print(gate.matrix)
        #         print()
        #     print("Decompose product")
        #     print(prod)
        return ans


def _zyzRule(gate):
    """ decomposition the unitary gate with 2 * 2 unitary into Rz Ry Rz sequence

    Args:
        gate(Unitary): the gate to be decomposed

    Returns:
        compositeGate: a list of compositeGate
    """
    unitary = gate.matrix
    targ = gate.targ
    eps = 1e-13

    det = linalg.det(unitary)
    beta_plus_delta = 0
    beta_dec_delta = 0
    if abs(det - 1) > eps:
        unitary[:] /= np.sqrt(det)

    if abs(unitary[0, 0]) > abs(unitary[0, 1]) > eps:
        gamma = _arccos((2 * (unitary[0, 0] * unitary[1, 1]).real - 1))
    else:
        gamma = _arccos((2 * (unitary[0, 1] * unitary[1, 0]).real + 1))
    if abs(unitary[0, 0]) > eps:
        beta_plus_delta = -np.angle(unitary[0, 0] / np.cos(gamma / 2)) * 2
    if abs(unitary[0, 1]) > eps:
        beta_dec_delta = np.angle(unitary[1, 0] / np.sin(gamma / 2)) * 2

    beta = (beta_plus_delta + beta_dec_delta) / 2
    delta = beta_plus_delta - beta
    compositeGate = CompositeGate()
    with compositeGate:
        if not _check2pi(delta):
            Rz(delta) & targ
        if not _check2pi(gamma):
            Ry(gamma) & targ
        if not _check2pi(beta):
            Rz(beta) & targ
    return compositeGate


ZyzRule = SU2TransformRule(_zyzRule)


def _xyxRule(gate):
    """ decomposition the unitary gate with 2 * 2 unitary into Rx Ry Rx sequence

    Args:
        gate(Unitary): the gate to be decomposed

    Returns:
        compositeGate: a list of compositeGate
    """
    unitary = gate.matrix
    targ = gate.targ
    eps = 1e-13
    unitary = np.array([
        [0.5 * (unitary[0, 0] + unitary[0, 1] + unitary[1, 0] + unitary[1, 1]),
         0.5 * (unitary[0, 0] - unitary[0, 1] + unitary[1, 0] - unitary[1, 1])],
        [0.5 * (unitary[0, 0] + unitary[0, 1] - unitary[1, 0] - unitary[1, 1]),
         0.5 * (unitary[0, 0] - unitary[0, 1] - unitary[1, 0] + unitary[1, 1])]
    ], dtype=np.complex128)
    det = linalg.det(unitary)
    beta_plus_delta = 0
    beta_dec_delta = 0
    if abs(det - 1) > eps:
        unitary[:] /= np.sqrt(det)

    if abs(unitary[0, 0]) > abs(unitary[0, 1]) > eps:
        gamma = _arccos((2 * (unitary[0, 0] * unitary[1, 1]).real - 1))
    else:
        gamma = _arccos((2 * (unitary[0, 1] * unitary[1, 0]).real + 1))
    if abs(unitary[0, 0]) > eps:
        beta_plus_delta = -np.angle(unitary[0, 0] / np.cos(gamma / 2)) * 2
    if abs(unitary[0, 1]) > eps:
        beta_dec_delta = np.angle(unitary[1, 0] / np.sin(gamma / 2)) * 2

    beta = (beta_plus_delta + beta_dec_delta) / 2
    delta = beta_plus_delta - beta
    compositeGate = CompositeGate()
    with compositeGate:
        if not _check2pi(delta):
            Rx(delta) & targ
        if not _check2pi(gamma):
            Ry(-gamma) & targ
        if not _check2pi(beta):
            Rx(beta) & targ
    return compositeGate


XyxRule = SU2TransformRule(_xyxRule)


def _ibmqRule(gate):
    unitary = gate.matrix
    targ = gate.targ
    eps = 1e-13

    det = linalg.det(unitary)
    beta_plus_delta = 0
    beta_dec_delta = 0
    if abs(det - 1) > eps:
        unitary[:] /= np.sqrt(det)

    if abs(unitary[0, 0]) > abs(unitary[0, 1]) > eps:
        gamma = _arccos((2 * (unitary[0, 0] * unitary[1, 1]).real - 1))
    else:
        gamma = _arccos((2 * (unitary[0, 1] * unitary[1, 0]).real + 1))
    if abs(unitary[0, 0]) > eps:
        beta_plus_delta = -np.angle(unitary[0, 0] / np.cos(gamma / 2)) * 2
    if abs(unitary[0, 1]) > eps:
        beta_dec_delta = np.angle(unitary[1, 0] / np.sin(gamma / 2)) * 2

    beta = (beta_plus_delta + beta_dec_delta) / 2
    delta = beta_plus_delta - beta
    compositeGate = CompositeGate()
    with compositeGate:
        if not _check2pi(delta):
            Rz(delta) & targ
        if not _check2pi(gamma):
            SX & targ
            Rz(gamma) & targ
            SX & targ
            X & targ
        if not _check2pi(beta):
            Rz(beta) & targ
    return compositeGate


IbmqRule = SU2TransformRule(_ibmqRule)


def _xzxRule(gate: BasicGate):
    unitary = gate.matrix
    det = linalg.det(unitary)
    targ = gate.targ
    eps = 1e-13
    if abs(det - 1) > eps:
        unitary[:] /= np.sqrt(det)

    delta1 = (unitary[0, 0] + unitary[1, 1]) ** 2 - (unitary[0, 1] + unitary[1, 0]) ** 2
    beta = _arccos((delta1.real - 2) / 2)
    alpha_plus_gamma_half = _arccos((unitary[0, 0] + unitary[1, 1]).real / (2 * np.cos(beta / 2)))
    alpha_minus_gamma_half = (_arccos((unitary[0, 0] - unitary[1, 1]) / (-2j * np.sin(beta / 2)))).real
    alpha = alpha_plus_gamma_half + alpha_minus_gamma_half
    gamma = alpha_plus_gamma_half - alpha_minus_gamma_half

    cg = CompositeGate()
    with cg:
        Rx(gamma) & targ
        Rz(beta) & targ
        Rx(alpha) & targ

    return cg


XzxRule = SU2TransformRule(_xzxRule)
