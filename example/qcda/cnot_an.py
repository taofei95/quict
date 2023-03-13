#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/27 6:35 下午
# @Author  : Han Yu
# @File    : CNOT_an.py
from QuICT.qcda.optimization import CnotAncilla
from QuICT.core import Circuit
from QuICT.core.gate import *


circuit = Circuit(4)
for _ in range(1000):
    for i in range(3):
        CX | circuit([i, i + 1])

CA = CnotAncilla(size=1)
result_circuit = CA.execute(circuit)
print(result_circuit)
