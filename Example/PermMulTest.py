#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/25 11:26 下午
# @Author  : Han Yu
# @File    : PermMulTest.py

from QuICT.models import *
from QuICT.algorithm import Amplitude

circuit = Circuit(2)
X             | circuit(0)
print(Amplitude.run(circuit))
X             | circuit(1)
print(Amplitude.run(circuit))
