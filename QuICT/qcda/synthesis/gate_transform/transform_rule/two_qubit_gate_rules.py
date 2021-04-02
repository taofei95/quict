#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu
# @File    : two_qubit_gate_rules.py

import math
from .transform_rule import TransformRule
from QuICT.core import *
from numpy import pi

"""
the file describe TransformRule between two kinds of 2-qubit gates.
"""


"""
transform CX gate into others
"""
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
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
    return gateSet
Cx2CrzRule = TransformRule(_cx2crz_rule, CX, CRz)

"""
transform CY gate into others
"""
def _cy2cx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S_dagger & targs[1]
        CX & [targs[0], targs[1]]
        S & targs[1]
    return gateSet
Cy2CxRule = TransformRule(_cy2cx_rule, CY, CX)

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

def _cy2crz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(math.pi / 2) & targs[0]
        S_dagger & targs[1]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        S & targs[1]
    return gateSet
Cy2CrzRule = TransformRule(_cy2crz_rule, CY, CRz)

"""
transform CZ gate into others
"""
def _cz2cx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        CX & [targs[0], targs[1]]
        H & targs[1]
    return gateSet
Cz2CxRule = TransformRule(_cz2cx_rule, CZ, CX)

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
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
    return gateSet
Cz2CrzRule = TransformRule(_cz2crz_rule, CZ, CRz)

"""
transform CRz gate into others
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

"""
transform CH gate into others
"""
def _ch2cx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        CX & targs
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
Ch2CxRule = TransformRule(_ch2cx_rule, CH, CX)

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

def _ch2crz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(math.pi / 2) & targs[0]
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
Ch2CrzRule = TransformRule(_ch2crz_rule, CH, CRz)

"""
transform Rxx gate into others
"""
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
Rxx2CxRule = TransformRule(_rxx2cx_rule, Rxx, CX)

def _rxx2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        H & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(theta) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        H & targs[1]
        H & targs[0]
    return gateSet
Rxx2CyRule = TransformRule(_rxx2cy_rule, Rxx, CY)

def _rxx2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        CZ & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[0]
    return gateSet
Rxx2CzRule = TransformRule(_rxx2cz_rule, Rxx, CZ)

def _rxx2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        H & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Rz(theta) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[1]
        H & targs[0]
    return gateSet
Rxx2ChRule = TransformRule(_rxx2ch_rule, Rxx, CH)

def _rxx2crz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[0]
    return gateSet
Rxx2CrzRule = TransformRule(_rxx2crz_rule, Rxx, CRz)

"""
transform Ryy gate into others
"""
def _ryy2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
    return gateSet
Ryy2CxRule = TransformRule(_ryy2cx_rule, Ryy, CX)

def _ryy2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(theta) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
    return gateSet
Ryy2CyRule = TransformRule(_ryy2cy_rule, Ryy, CY)

def _ryy2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
    return gateSet
Ryy2CzRule = TransformRule(_ryy2cz_rule, Ryy, CZ)

def _ryy2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Rz(theta) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
    return gateSet
Ryy2ChRule = TransformRule(_ryy2ch_rule, Ryy, CH)

def _ryy2crz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
    return gateSet
Ryy2CrzRule = TransformRule(_ryy2crz_rule, Ryy, CRz)

"""
transform Rzz gate into others
"""
def _rzz2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
    return gateSet 
Rzz2CxRule = TransformRule(_rzz2cx_rule, Rzz, CX)

def _rzz2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(theta) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
    return gateSet
Rzz2CyRule = TransformRule(_rzz2cy_rule, Rzz, CY)

def _rzz2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
    return gateSet 
Rzz2CzRule = TransformRule(_rzz2cz_rule, Rzz, CZ)

def _rzz2ch_rule(gate):
    theta = gate.pargs[0]
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
        Rz(theta) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
    return gateSet 
Rzz2ChRule = TransformRule(_rzz2ch_rule, Rzz, CH)

def _rzz2crz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
    return gateSet 
Rzz2CrzRule = TransformRule(_rzz2crz_rule, Rzz, CRz)
