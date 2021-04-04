#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu    Dang Haoran    Cai CHao
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
CX2CYRule = TransformRule(_cx2cy_rule, CX, CY)

def _cx2cz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        CZ & targs
        H & targs[1]
    return gateSet
CX2CZRule = TransformRule(_cx2cz_rule, CX, CZ)

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
CX2CHRule = TransformRule(_cx2ch_rule, CX, CH)

def _cx2crz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
    return gateSet
CX2CRzRule = TransformRule(_cx2crz_rule, CX, CRz)

def _cx2rxx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
    return gateSet
CX2RxxRule = TransformRule(_cx2rxx_rule, CX, Rxx)

def _cx2ryy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Ryy(math.pi / 2) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
    return gateSet
CX2RyyRule = TransformRule(_cx2ryy_rule, CX, Ryy)

def _cx2rzz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rzz(math.pi / 2) & targs
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
    return gateSet
CX2RzzRule = TransformRule(_cx2rzz_rule, CX, Rzz)

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
CY2CXRule = TransformRule(_cy2cx_rule, CY, CX)

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
CY2CZRule = TransformRule(_cy2cz_rule, CY, CZ)

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
CY2CHRule = TransformRule(_cy2ch_rule, CY, CH)

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
CY2CRzRule = TransformRule(_cy2crz_rule, CY, CRz)

def _cy2rxx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S_dagger & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        S & targs[1]
    return gateSet
CY2RxxRule = TransformRule(_cy2rxx_rule, CY, Rxx)

def _cy2ryy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S_dagger & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Ryy(math.pi / 2) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        S & targs[1]
    return gateSet
CY2RyyRule = TransformRule(_cy2ryy_rule, CY, Ryy)

def _cy2rzz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S_dagger & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rzz(math.pi / 2) & targs
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        S & targs[1]
    return gateSet
CY2RzzRule = TransformRule(_cy2rzz_rule, CY, Rzz)

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
CZ2CXRule = TransformRule(_cz2cx_rule, CZ, CX)

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
CZ2CYRule = TransformRule(_cz2cy_rule, CZ, CY)

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
CZ2CHRule = TransformRule(_cz2ch_rule, CZ, CH)

def _cz2crz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
    return gateSet
CZ2CRzRule = TransformRule(_cz2crz_rule, CZ, CRz)

def _cz2rxx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        H & targs[1]
    return gateSet
CZ2RxxRule = TransformRule(_cz2rxx_rule, CZ, Rxx)

def _cz2ryy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Ryy(math.pi / 2) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        H & targs[1]
    return gateSet
CZ2RyyRule = TransformRule(_cz2ryy_rule, CZ, Ryy)

def _cz2rzz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rzz(math.pi / 2) & targs
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        H & targs[1]
    return gateSet
CZ2RzzRule = TransformRule(_cz2rzz_rule, CZ, Rzz)

"""
transform CRz gate into others
"""
def _crz2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        """decomposition1:"""
        Rz(theta / 2) & targs[1]
        CX & targs
        Rz(-theta / 2) & targs[1]
        CX & targs
        """decomposition2:
        U1(theta / 2) & targs[1]
        CX & targs
        U1(-theta / 2) & targs[1]
        CX & targs
        """
    return gateSet
CRz2CXRule = TransformRule(_crz2cx_rule, CRz, CX)

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
CRz2CYRule = TransformRule(_crz2cy_rule, CRz, CY)

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
CRz2CZRule = TransformRule(_crz2cz_rule, CRz, CZ)

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
CRz2CHRule = TransformRule(_crz2ch_rule, CRz, CH)

def _crz2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        U1(theta / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        U1(-theta / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
    return gateSet
CRz2RxxRule = TransformRule(_crz2rxx_rule, CRz, Rxx)

def _crz2ryy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        U1(theta / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Ryy(math.pi / 2) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        U1(-theta / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Ryy(math.pi / 2) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
    return gateSet
CRz2RyyRule = TransformRule(_crz2ryy_rule, CRz, Ryy)

def _crz2rzz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        U1(theta / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rzz(math.pi / 2) & targs
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        U1(-theta / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rzz(math.pi / 2) & targs
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
    return gateSet
CRz2RzzRule = TransformRule(_crz2rzz_rule, CRz, Rzz)

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
CH2CXRule = TransformRule(_ch2cx_rule, CH, CX)

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
CH2CYRule = TransformRule(_ch2cy_rule, CH, CY)

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
CH2CZRule = TransformRule(_ch2cz_rule, CH, CZ)

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
CH2CRzRule = TransformRule(_ch2crz_rule, CH, CRz)

def _ch2rxx_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
CH2RxxRule = TransformRule(_ch2rxx_rule, CH, Rxx)

def _ch2ryy_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Ryy(math.pi / 2) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
CH2RyyRule = TransformRule(_ch2ryy_rule, CH, Ryy)

def _ch2rzz_rule(gate):
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rzz(math.pi / 2) & targs
        H & targs[1]
        H & targs[0]
        Ry(-math.pi / 2) & targs[0]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return gateSet
CH2RzzRule = TransformRule(_ch2rzz_rule, CH, Rzz)

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
Rxx2CXRule = TransformRule(_rxx2cx_rule, Rxx, CX)

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
Rxx2CYRule = TransformRule(_rxx2cy_rule, Rxx, CY)

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
Rxx2CZRule = TransformRule(_rxx2cz_rule, Rxx, CZ)

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
Rxx2CHRule = TransformRule(_rxx2ch_rule, Rxx, CH)

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
Rxx2CRzRule = TransformRule(_rxx2crz_rule, Rxx, CRz)

def _rxx2ryy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Ryy(theta) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
    return gateSet
Rxx2RyyRule = TransformRule(_rxx2ryy_rule, Rxx, Ryy)

def _rxx2rzz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        H & targs[0]
        H & targs[1]
        Rzz(theta) & targs
        H & targs[1]
        H & targs[0]
    return gateSet
Rxx2RzzRule = TransformRule(_rxx2rzz_rule, Rxx, Rzz)

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
Ryy2CXRule = TransformRule(_ryy2cx_rule, Ryy, CX)

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
Ryy2CYRule = TransformRule(_ryy2cy_rule, Ryy, CY)

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
Ryy2CZRule = TransformRule(_ryy2cz_rule, Ryy, CZ)

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
Ryy2CHRule = TransformRule(_ryy2ch_rule, Ryy, CH)

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
Ryy2CRzRule = TransformRule(_ryy2crz_rule, Ryy, CRz)

def _ryy2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        """Decomposition1:"""
        Rx(math.pi / 2) & targs[0]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rxx(math.pi / 2) & targs
        Rz(theta) & targs[1]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        """Decomposition2:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rxx(theta) & targs
        H & targs[1]
        H & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        """
    return gateSet
Ryy2RxxRule = TransformRule(_ryy2rxx_rule, Ryy, Rxx)

def _ryy2rzz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        Rzz(theta) & targs
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
    return gateSet
Ryy2RzzRule = TransformRule(_ryy2rzz_rule, Ryy, Rzz)

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
Rzz2CXRule = TransformRule(_rzz2cx_rule, Rzz, CX)

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
Rzz2CYRule = TransformRule(_rzz2cy_rule, Rzz, CY)

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
Rzz2CZRule = TransformRule(_rzz2cz_rule, Rzz, CZ)

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
Rzz2CHRule = TransformRule(_rzz2ch_rule, Rzz, CH)

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
Rzz2CRzRule = TransformRule(_rzz2crz_rule, Rzz, CRz)

def _rzz2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        Rz(theta) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
    return gateSet
Rzz2RxxRule = TransformRule(_rzz2rxx_rule, Rzz, Rxx)

def _rzz2ryy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.affectArgs
    gateSet = GateSet()
    with gateSet:
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Ryy(theta) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
    return gateSet
Rzz2RyyRule = TransformRule(_rzz2ryy_rule, Rzz, Ryy)