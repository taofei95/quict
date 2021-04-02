#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu
# @File    : two_qubit_gate_rules.py

from .transform_rule import TransformRule
from QuICT.core import *
from numpy import pi

"""

the file describe TransformRule between two kinds of 2-qubit gates.

"""


def _crz2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(theta / 2) & targs[1]
        CX & targs
        Rz(-theta / 2) & targs[1]
        CX & targs
    return gateSet
Crz2CxRule = TransformRule(_crz2cx_rule, CRz, CX)

#错误
def _ryy2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(pi / 2) & targs[0]
        Rx(pi / 2) & targs[1]
        H & targs[1]
        Rxx(theta) & targs
        H & targs[0]
        H & targs[1]
        Rx(-pi / 2) & targs[0]
        Rx(-pi / 2) & targs[1]
    return gateSet
Ryy2RxxRule = TransformRule(_ryy2rxx_rule, Ryy, Rxx)

def _cz2cx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        CX & targs
        H & targs[1]
    return gateSet
CZ2CXRule = TransformRule(_cz2cx_rule, CZ, CX)

def _rxx2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        H & targs[1]
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
        H & targs[1]
        H & targs[0]
    return gateSet
Rxx2CXRule = TransformRule(_rxx2cx_rule, Rxx, CX)