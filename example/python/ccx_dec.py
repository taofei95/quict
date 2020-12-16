#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/6 10:04 下午
# @Author  : Han Yu
# @File    : CCX_Dec.py

from QuICT.core import *
from QuICT.algorithm import Amplitude

circuit = Circuit(3)

X        | circuit(0)
H        | circuit(1)
H        | circuit(2)
CX       | circuit([1, 0])
T_dagger | circuit(0)
CX       | circuit([2, 0])
T        | circuit(0)
CX       | circuit([1, 0])
T_dagger | circuit(0)
CX       | circuit([2, 0])
T        | circuit(0)
T_dagger | circuit(1)
H        | circuit(0)
CX       | circuit([2, 1])
T_dagger | circuit(1)
CX       | circuit([2, 1])
T        | circuit(2)
S        | circuit(1)
H        | circuit(2)
H        | circuit(1)
X        | circuit(2)
X        | circuit(1)
H        | circuit(1)
CX       | circuit([2, 1])
H        | circuit(1)
X        | circuit(2)
X        | circuit(1)
H        | circuit(2)
H        | circuit(1)

amplitude = Amplitude.run(circuit)
print(amplitude)
