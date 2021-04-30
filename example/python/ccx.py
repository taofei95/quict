#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/6 10:04 下午
# @Author  : Han Yu
# @File    : CCX_Dec.py

from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary, Amplitude

circuit = Circuit(3)

"""
H           | circuit(2)
CX          | circuit([2, 1])
T_dagger    | circuit(1)
CX          | circuit([0, 1])
T           | circuit(1)
CX          | circuit([2, 1])
T_dagger    | circuit(1)
CX          | circuit([0, 1])
T           | circuit(1)
CX          | circuit([0, 2])
T_dagger    | circuit(2)
CX          | circuit([0, 2])
T           | circuit(0)
T           | circuit(2)
H           | circuit(2)
"""
Unitary([
    1, 0,
    0, 1
]) | circuit

Phase(0) | circuit
circuit.draw_photo(show_depth=False)

unitary = SyntheticalUnitary.run(circuit, showSU=False)
print(unitary)
