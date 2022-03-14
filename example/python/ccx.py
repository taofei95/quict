#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/6 10:04 下午
# @Author  : Han Yu
# @File    : CCX_Dec.py

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.algorithm import SyntheticalUnitary


circuit = Circuit(3)

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

unitary = SyntheticalUnitary.run(circuit, showSU=False)
print(unitary)
