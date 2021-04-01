#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu
# @File    : two_qubit_gate_rules.py

import math
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

def _cx2cy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
    return gateSet
Cx2CyRule = TransformRule(_cx2cy_rule, CX, CY)

def _cx2cz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        CZ & targs
        H & targs[1]
    return gateSet
Cx2CzRule = TransformRule(_cx2cz_rule, CX, CZ)

def _cx2ch_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
    return gateSet
Cx2ChRule = TransformRule(_cx2ch_rule, CX, CH)

def _cx2crz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Phase(math.pi/2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
    return gateSet
Cx2CrzRule = TransformRule(_cx2crz_rule, CX, CRz)

def _cy2cz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S_dagger & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        S & targs[1]
    return gateSet
Cy2CzRule = TransformRule(_cy2cz_rule, CY, CZ)

def _cy2ch_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S_dagger & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        S & targs[1]
    return gateSet
Cy2ChRule = TransformRule(_cy2ch_rule, CY, CH)

def _cz2cy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        H & targs[1]
    return gateSet
Cz2CyRule = TransformRule(_cz2cy_rule, CZ, CY)

def _cz2ch_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[1]
    return gateSet
Cz2ChRule = TransformRule(_cz2ch_rule, CZ, CH)

def _cz2crz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Phase(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
    return gateSet
Cz2CrzRule = TransformRule(_cz2crz_rule, CZ, CRz)

def _crz2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(theta / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(-theta / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
    return gateSet
Crz2CyRule = TransformRule(_crz2cy_rule, CRz, CY)

def _crz2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(theta / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rz(-theta / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
    return gateSet
Crz2CzRule = TransformRule(_crz2cz_rule, CRz, CZ)

def _crz2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(theta / 2) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Rz(-theta / 2) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
    return gateSet
Crz2ChRule = TransformRule(_crz2ch_rule, CRz, CH)

def _ch2cy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
Ch2CyRule = TransformRule(_ch2cy_rule, CH, CY)

def _ch2cz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
Ch2CzRule = TransformRule(_ch2cz_rule, CH, CZ)

def _rxx2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
       # H & targs[0]
        H & targs[1]
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
        H & targs[1]
        H & targs[0]
    return gateSet
Rxx2CxRule = TransformRule(_rxx2cx_rule, Rxx, CX)