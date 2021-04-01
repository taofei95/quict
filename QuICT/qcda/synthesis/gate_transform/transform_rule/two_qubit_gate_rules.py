#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu
# @File    : two_qubit_gate_rules.py

from .transform_rule import TransformRule
from QuICT.core import *

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
