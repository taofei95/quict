#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/8 3:29 下午
# @Author  : Han Yu
# @File    : Rzz.py

import numpy as np

from QuICT import *
from QuICT.algorithm import *

circuit = Circuit(3)
X | circuit(1)
PermFx([0, 0, 1, 0]) | circuit
print(PermFx.pargs)
amplitude = Amplitude.run(circuit)
print(amplitude)
