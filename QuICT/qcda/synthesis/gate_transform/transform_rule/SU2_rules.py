#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:42 下午
# @Author  : Han Yu
# @File    : SU2_rules.py

"""

the file describe TransformRule the decomposite SU(2) into instruction set.

"""

import numpy as np
from numpy import arccos, angle, exp, linalg
from scipy.stats import ortho_group

from QuICT.core import *

from .transform_rule import TransformRule

class SU2TransformRule(TransformRule):
    """ a subClass of TransformRule, to check the decomposition of SU2

    """
    def check_equal(self, ignore_phase = True, eps = 1e-13):
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
        gateSet = self.transform(gate)
        ans = gateSet.equal(gate, ignore_phase=ignore_phase, eps=eps)
        if not ans:
            print(U)
        return ans

def _zyzRule(gate):
    """ decomposition the unitary gate with 2 * 2 unitary into Rz Ry Rz sequence
    Args:
        gate(Unitary): the gate to be decomposed

    Returns:
        gateSet: a list of gateSet
    """
    unitary = gate.matrix
    targ = gate.targ
    eps = 1e-13
    
    det = linalg.det(unitary)
    gamma = 0
    beta_plus_delta = 0
    beta_dec_delta = 0
    if abs(det - 1) > eps:
        unitary[:] /= np.sqrt(det)
    print(unitary)
    if abs(unitary[0, 0]) > eps:
        gamma = arccos((2 * abs(unitary[0, 0] * unitary[1, 1]) - 1))
        beta_plus_delta = -np.angle(unitary[0, 0] / np.cos(gamma / 2)) * 2
    if abs(unitary[0, 1]) > eps:
        gamma = arccos((2 * -abs(unitary[0, 1] * unitary[1, 0]) + 1))
        beta_dec_delta = np.angle(unitary[1, 0] / np.sin(gamma / 2)) * 2
    beta = (beta_plus_delta + beta_dec_delta) / 2
    delta = beta_plus_delta - beta
    gateSet = GateSet()
    with gateSet:
        Rz(delta)  & targ
        Ry(gamma) & targ
        Rz(beta) & targ
    return gateSet

ZyzRule = SU2TransformRule(_zyzRule)

def _xyxRule(gate):
    pass
ZxzRule = SU2TransformRule(_xyxRule)

def _ibmqRule(gate):
    unitary = gate.matrix
    targ = gate.targ
    eps = 1e-13

    det = linalg.det(unitary)
    gamma = 0
    beta_plus_delta = 0
    beta_dec_delta = 0
    if abs(det - 1) > eps:
        unitary[:] /= np.sqrt(det)
    if abs(unitary[0, 0]) > eps:
        gamma = arccos((2 * unitary[0, 0] * unitary[1, 1] - 1).real)
        beta_plus_delta = -np.angle(unitary[0, 0] / np.cos(gamma / 2)) * 2
    if abs(unitary[0, 1]) > eps:
        gamma = arccos((2 * unitary[0, 1] * unitary[1, 0] + 1).real)
        beta_dec_delta = np.angle(unitary[1, 0] / np.sin(gamma / 2)) * 2
    beta = (beta_plus_delta + beta_dec_delta) / 2
    delta = beta_plus_delta - beta
    gateSet = GateSet()
    with gateSet:
        Rz(delta)  & targ
        SX & targ
        Rz(gamma) & targ
        SX & targ
        X & targ
        Rz(beta) & targ
    return gateSet
IbmqRule = SU2TransformRule(_ibmqRule)