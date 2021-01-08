#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/8 3:29 下午
# @Author  : Han Yu
# @File    : Rzz.py

import numpy as np

from QuICT import *
from QuICT.algorithm import *

circuit = Circuit(2)
H | circuit
RZZ(np.pi / 2) | circuit
Phase(np.pi / 4) | circuit(0)
amplitude = Amplitude.run(circuit)
print(amplitude)
