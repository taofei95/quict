#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu    Dang Haoran    Cai CHao
# @File    : two_qubit_gate_rules.py

import math
from .transform_rule import TransformRule
from QuICT.core.gate import *

"""
the file describe TransformRule between two kinds of 2-qubit gates.
"""

"""
transform CX gate into others
"""


def _cx2cy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
    return compositeGate


Cx2CyRule = TransformRule(_cx2cy_rule, CX, CY)


def _cx2cz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        CZ & targs
        H & targs[1]
    return compositeGate


Cx2CzRule = TransformRule(_cx2cz_rule, CX, CZ)


def _cx2ch_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
    return compositeGate


Cx2ChRule = TransformRule(_cx2ch_rule, CX, CH)


def _cx2crz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
    return compositeGate


Cx2CrzRule = TransformRule(_cx2crz_rule, CX, CRz)


def _cx2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        FSim(0, math.pi) & targs
        H & targs[1]
    return compositeGate


Cx2FsimRule = TransformRule(_cx2fsim_rule, CX, FSim)


def _cx2rxx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
    return compositeGate


Cx2RxxRule = TransformRule(_cx2rxx_rule, CX, Rxx)


def _cx2ryy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Cx2RyyRule = TransformRule(_cx2ryy_rule, CX, Ryy)


def _cx2rzz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Cx2RzzRule = TransformRule(_cx2rzz_rule, CX, Rzz)


"""
transform CY gate into others
"""


def _cy2cx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S_dagger & targs[1]
        CX & [targs[0], targs[1]]
        S & targs[1]
    return compositeGate


Cy2CxRule = TransformRule(_cy2cx_rule, CY, CX)


def _cy2cz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S_dagger & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        S & targs[1]
    return compositeGate


Cy2CzRule = TransformRule(_cy2cz_rule, CY, CZ)


def _cy2ch_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S_dagger & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        S & targs[1]
    return compositeGate


Cy2ChRule = TransformRule(_cy2ch_rule, CY, CH)


def _cy2crz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(math.pi / 2) & targs[0]
        S_dagger & targs[1]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        S & targs[1]
    return compositeGate


Cy2CrzRule = TransformRule(_cy2crz_rule, CY, CRz)


def _cy2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S_dagger & targs[1]
        H & targs[1]
        FSim(0, math.pi) & targs
        H & targs[1]
        S & targs[1]
    return compositeGate


Cy2FsimRule = TransformRule(_cy2fsim_rule, CY, FSim)


def _cy2rxx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S_dagger & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        S & targs[1]
    return compositeGate


Cy2RxxRule = TransformRule(_cy2rxx_rule, CY, Rxx)


def _cy2ryy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Cy2RyyRule = TransformRule(_cy2ryy_rule, CY, Ryy)


def _cy2rzz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Cy2RzzRule = TransformRule(_cy2rzz_rule, CY, Rzz)


"""
transform CZ gate into others
"""


def _cz2cx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        CX & [targs[0], targs[1]]
        H & targs[1]
    return compositeGate


Cz2CxRule = TransformRule(_cz2cx_rule, CZ, CX)


def _cz2cy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        H & targs[1]
    return compositeGate


Cz2CyRule = TransformRule(_cz2cy_rule, CZ, CY)


def _cz2ch_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[1]
    return compositeGate


Cz2ChRule = TransformRule(_cz2ch_rule, CZ, CH)


def _cz2crz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
    return compositeGate


Cz2CrzRule = TransformRule(_cz2crz_rule, CZ, CRz)


def _cz2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        FSim(0, math.pi) & targs
    return compositeGate


Cz2FsimRule = TransformRule(_cz2fsim_rule, CZ, FSim)


def _cz2rxx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        H & targs[1]
    return compositeGate


Cz2RxxRule = TransformRule(_cz2rxx_rule, CZ, Rxx)


def _cz2ryy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Cz2RyyRule = TransformRule(_cz2ryy_rule, CZ, Ryy)


def _cz2rzz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Cz2RzzRule = TransformRule(_cz2rzz_rule, CZ, Rzz)


"""
transform CRz gate into others
"""


def _crz2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Crz2CxRule = TransformRule(_crz2cx_rule, CRz, CX)


def _crz2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(theta / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(-theta / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
    return compositeGate


Crz2CyRule = TransformRule(_crz2cy_rule, CRz, CY)


def _crz2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(theta / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rz(-theta / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
    return compositeGate


Crz2CzRule = TransformRule(_crz2cz_rule, CRz, CZ)


def _crz2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Crz2ChRule = TransformRule(_crz2ch_rule, CRz, CH)


def _crz2fsim_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(theta / 2) & targs[1]
        H & targs[1]
        FSim(0, np.pi) & targs
        Rx(-theta / 2) & targs[1]
        FSim(0, np.pi) & targs
        H & targs[1]
    return compositeGate


Crz2FsimRule = TransformRule(_crz2fsim_rule, CRz, FSim)


def _crz2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Crz2RxxRule = TransformRule(_crz2rxx_rule, CRz, Rxx)


def _crz2ryy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Crz2RyyRule = TransformRule(_crz2ryy_rule, CRz, Ryy)


def _crz2rzz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Crz2RzzRule = TransformRule(_crz2rzz_rule, CRz, Rzz)


"""
transform CH gate into others
"""


def _ch2cx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        CX & targs
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return compositeGate


Ch2CxRule = TransformRule(_ch2cx_rule, CH, CX)


def _ch2cy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return compositeGate


Ch2CyRule = TransformRule(_ch2cy_rule, CH, CY)


def _ch2cz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
    return compositeGate


Ch2CzRule = TransformRule(_ch2cz_rule, CH, CZ)


def _ch2crz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ch2CrzRule = TransformRule(_ch2crz_rule, CH, CRz)


def _ch2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        Rx(math.pi / 4) & targs[1]
        FSim(0, math.pi) & targs
        Rx(-math.pi / 4) & targs[1]
        S_dagger & targs[1]
    return compositeGate


Ch2FsimRule = TransformRule(_ch2fsim_rule, CH, FSim)


def _ch2rxx_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ch2RxxRule = TransformRule(_ch2rxx_rule, CH, Rxx)


def _ch2ryy_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ch2RyyRule = TransformRule(_ch2ryy_rule, CH, Ryy)


def _ch2rzz_rule(gate):
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ch2RzzRule = TransformRule(_ch2rzz_rule, CH, Rzz)


"""
transform Rxx gate into others
"""


def _rxx2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        H & targs[1]
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
        H & targs[1]
        H & targs[0]
    return compositeGate


Rxx2CxRule = TransformRule(_rxx2cx_rule, Rxx, CX)


def _rxx2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Rxx2CyRule = TransformRule(_rxx2cy_rule, Rxx, CY)


def _rxx2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        CZ & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[0]
    return compositeGate


Rxx2CzRule = TransformRule(_rxx2cz_rule, Rxx, CZ)


def _rxx2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Rxx2ChRule = TransformRule(_rxx2ch_rule, Rxx, CH)


def _rxx2crz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[0]
    return compositeGate


Rxx2CrzRule = TransformRule(_rxx2crz_rule, Rxx, CRz)


def _rxx2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    theta = gate.pargs[0]
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        FSim(0, math.pi) & targs
        Rx(theta) & targs[1]
        FSim(0, math.pi) & targs
        H & targs[0]
    return compositeGate


Rxx2FsimRule = TransformRule(_rxx2fsim_rule, Rxx, FSim)


def _rxx2ryy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Ryy(theta) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
    return compositeGate


Rxx2RyyRule = TransformRule(_rxx2ryy_rule, Rxx, Ryy)


def _rxx2rzz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        H & targs[1]
        Rzz(theta) & targs
        H & targs[1]
        H & targs[0]
    return compositeGate


Rxx2RzzRule = TransformRule(_rxx2rzz_rule, Rxx, Rzz)


"""
transform Ryy gate into others
"""


def _ryy2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
    return compositeGate


Ryy2CxRule = TransformRule(_ryy2cx_rule, Ryy, CX)


def _ryy2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ryy2CyRule = TransformRule(_ryy2cy_rule, Ryy, CY)


def _ryy2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ryy2CzRule = TransformRule(_ryy2cz_rule, Ryy, CZ)


def _ryy2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ryy2ChRule = TransformRule(_ryy2ch_rule, Ryy, CH)


def _ryy2crz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ryy2CrzRule = TransformRule(_ryy2crz_rule, Ryy, CRz)


def _ryy2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    theta = gate.pargs[0]
    compositeGate = CompositeGate()
    with compositeGate:
        Rx(np.pi / 2) & targs[0]
        Rx(np.pi / 2) & targs[1]
        H & targs[1]
        FSim(0, math.pi) & targs
        Rx(theta) & targs[1]
        FSim(0, math.pi) & targs
        H & targs[1]
        Rx(-np.pi / 2) & targs[0]
        Rx(-np.pi / 2) & targs[1]
    return compositeGate


Ryy2FsimRule = TransformRule(_ryy2fsim_rule, Ryy, FSim)


def _ryy2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Ryy2RxxRule = TransformRule(_ryy2rxx_rule, Ryy, Rxx)


def _ryy2rzz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        Rzz(theta) & targs
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
    return compositeGate


Ryy2RzzRule = TransformRule(_ryy2rzz_rule, Ryy, Rzz)


"""
transform Rzz gate into others
"""


def _rzz2cx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        CX & targs
        Rz(theta) & targs[1]
        CX & targs
    return compositeGate


Rzz2CxRule = TransformRule(_rzz2cx_rule, Rzz, CX)


def _rzz2cy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(theta) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
    return compositeGate


Rzz2CyRule = TransformRule(_rzz2cy_rule, Rzz, CY)


def _rzz2cz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
    return compositeGate


Rzz2CzRule = TransformRule(_rzz2cz_rule, Rzz, CZ)


def _rzz2ch_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Rzz2ChRule = TransformRule(_rzz2ch_rule, Rzz, CH)


def _rzz2crz_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
        Rz(theta) & targs[1]
        H & targs[1]
        Rz(math.pi / 2) & targs[0]
        CRz(math.pi) & targs
        H & targs[1]
    return compositeGate


Rzz2CrzRule = TransformRule(_rzz2crz_rule, Rzz, CRz)


def _rzz2fsim_rule(gate):
    targs = gate.cargs + gate.targs
    theta = gate.pargs[0]
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        FSim(0, math.pi) & targs
        Rx(theta) & targs[1]
        FSim(0, math.pi) & targs
        H & targs[1]
    return compositeGate


Rzz2FsimRule = TransformRule(_rzz2fsim_rule, Rzz, FSim)


def _rzz2rxx_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
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
    return compositeGate


Rzz2RxxRule = TransformRule(_rzz2rxx_rule, Rzz, Rxx)


def _rzz2ryy_rule(gate):
    theta = gate.pargs[0]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Ryy(theta) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
    return compositeGate


Rzz2RyyRule = TransformRule(_rzz2ryy_rule, Rzz, Ryy)


"""
transform Fsim gate into others
"""


def _fsim2cx_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        CX & targs
        H & targs[0]
        CX & [targs[1], targs[0]]
        Rz(-theta) & targs[0]
        CX & [targs[1], targs[0]]
        Rz(theta) & targs[0]
        Phase(theta / 2) & targs[0]
        H & targs[0]
        CX & targs
        U1(-fai / 2) & targs[1]
        CX & targs
        U1(fai / 2) & targs[1]
        CX & targs
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2CxRule = TransformRule(_fsim2cx_rule, FSim, CX)


def _fsim2cy_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        H & targs[0]
        S & targs[0]
        CY & [targs[1], targs[0]]
        Rz(-theta) & targs[0]
        CY & [targs[1], targs[0]]
        Rz(theta - math.pi / 2) & targs[0]
        H & targs[0]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        U1(-fai / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        U1(fai / 2) & targs[1]
        S & targs[1]
        CY & targs
        S_dagger & targs[1]
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2CyRule = TransformRule(_fsim2cy_rule, FSim, CY)


def _fsim2cz_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[1]
        CZ & targs
        H & targs[1]
        CZ & [targs[1], targs[0]]
        Rx(-theta) & targs[0]
        CZ & [targs[1], targs[0]]
        Rx(theta) & targs[0]
        H & targs[1]
        CZ & targs
        H & targs[1]
        U1(-fai / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        U1(fai / 2) & targs[1]
        H & targs[1]
        CZ & targs
        H & targs[1]
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2CzRule = TransformRule(_fsim2cz_rule, FSim, CZ)


def _fsim2ch_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        H & targs[0]
        T_dagger & targs[0]
        H & targs[0]
        S_dagger & targs[0]
        CH & [targs[1], targs[0]]
        Ry(theta) & targs[0]     
        CH & [targs[1], targs[0]]
        S & targs[0]       
        Rx(theta + math.pi / 4) & targs[0]
        T & targs[1]
        Rx(math.pi / 2) & targs[1]
        CH & targs
        Ry(fai / 2) & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        U1(-fai / 2) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        U1(fai / 2) & targs[1]
        T_dagger & targs[1]
        H & targs[1]
        S_dagger & targs[1]
        CH & targs
        S & targs[1]
        H & targs[1]
        T & targs[1]
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2ChRule = TransformRule(_fsim2ch_rule, FSim, CH)


def _fsim2crz_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        H & targs[0]
        Rz(math.pi / 2) & targs[1]
        CRz(math.pi) & [targs[1], targs[0]]
        Rz(math.pi / 2) & targs[1]
        Rx(-theta) & targs[0]
        CRz(math.pi) & [targs[1], targs[0]]
        Rx(theta) & targs[0]
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        U1(-fai / 2) & targs[1]
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        U1(fai / 2) & targs[1]
        Rz(math.pi / 2) & targs[0]
        H & targs[1]
        CRz(math.pi) & targs
        H & targs[1]
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2CrzRule = TransformRule(_fsim2crz_rule, FSim, CRz)


def _fsim2rxx_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        Rxx(theta) & targs
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        H & targs[0]
        H & targs[1]
        Rxx(theta) & targs
        H & targs[1]
        H & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        U1(-fai / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        U1(fai / 2) & targs[1]
        Ry(math.pi / 2) & targs[0]
        Phase(7 * math.pi / 4) & targs[0]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Rxx(math.pi / 2) & targs
        Ry(-math.pi / 2) & targs[0]
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2RxxRule = TransformRule(_fsim2rxx_rule, FSim, Rxx)


def _fsim2ryy_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        H & targs[1]
        Rx(-math.pi / 2) & targs[0]
        Rx(-math.pi / 2) & targs[1]
        Ryy(theta) & targs
        Rx(math.pi / 2) & targs[1]
        Rx(math.pi / 2) & targs[0]
        H & targs[1]
        H & targs[0]
        Ryy(theta) & targs
        U1(-fai / 2) & targs[1]
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
        U1(fai / 2) & targs[1]
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
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2RyyRule = TransformRule(_fsim2ryy_rule, FSim, Ryy)


def _fsim2rzz_rule(gate):
    theta = gate.pargs[0]
    fai = gate.pargs[1]
    targs = gate.cargs + gate.targs
    compositeGate = CompositeGate()
    with compositeGate:
        H & targs[0]
        H & targs[1]
        Rzz(theta) & targs
        H & targs[1]
        H & targs[0]
        Rx(math.pi / 2) & targs[0]
        Rx(math.pi / 2) & targs[1]
        Rzz(theta) & targs
        Rx(-math.pi / 2) & targs[1]
        Rx(-math.pi / 2) & targs[0]
        U1(-fai / 2) & targs[1]
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
        U1(fai / 2) & targs[1]
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
        Rz(-fai / 2) & targs[0]
    return compositeGate


Fsim2RzzRule = TransformRule(_fsim2rzz_rule, FSim, Rzz)
