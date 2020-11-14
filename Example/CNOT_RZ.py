#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/24 9:44 上午
# @Author  : Han Yu
# @File    : CNOT_RZ.py

from QuICT.algorithm import CNOT_RZ, SyntheticalUnitary
from QuICT.models import *
from math import pi

circuit = Circuit(4)
CX         | circuit([0, 1])
Rz(pi / 8) | circuit(1)
CX         | circuit([1, 2])
Rz(pi / 4) | circuit(2)
CX         | circuit([2, 3])
Rz(pi / 8) | circuit(3)
circuit.draw_photo()
circuit.print_infomation()
result = CNOT_RZ.run(circuit)
result.print_infomation()
result.draw_photo()



