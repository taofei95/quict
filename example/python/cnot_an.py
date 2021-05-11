#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/27 6:35 下午
# @Author  : Han Yu
# @File    : CNOT_an.py

from QuICT.qcda.optimization import CnotAncillae
from QuICT.core import *

for n in range(4, 5):
    circuit = Circuit(n)
    for _ in range(50):
        for i in range(n - 1):
            CX | circuit([i, i + 1])
    result_circuit = CnotAncillae.run(circuit, size = 1)
    result_circuit.print_infomation()
    result_circuit.draw()
