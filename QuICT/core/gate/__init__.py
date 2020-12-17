#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/10 11:09
# @Author  : Han Yu
# @File    : __init__.py

from .gate import (
    H,
    S,
    S_dagger,
    X,
    Y,
    Z,
    ID,
    U1,
    U2,
    U3,
    Rx,
    Ry,
    Rz,
    T,
    T_dagger,
    CZ,
    CX,
    CY,
    CH,
    CRz,
    CCX,
    Measure,
    Reset,
    Barrier,
    Swap,
    Perm,
    PermShift,
    ControlPermShift,
    PermMul,
    ControlPermMul,
    PermFx,
    Unitary,
    ControlPermMulDetail,
    ShorInitial,

    BasicGate,

    GATE_ID,
    GATE_REGISTER
)

from .gate_builder import (
    GateBuilder
)

from .extension_gate import (
    gateModel,

    QFT,
    IQFT,
    RZZ,
    CU1,
    CU3,
    CRz_Decompose,
    CCX_Decompose,
    CCRz,
    Fredkin,

    EXTENSION_GATE_ID,
    EXTENSION_GATE_REGISTER
)
