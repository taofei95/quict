#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/12 11:20 上午
# @Author  : Han Yu
# @File    : Perm.py

import numpy as np

from QuICT import *
from QuICT.algorithm import *

circuit = Circuit(4)
circuit.assign_initial_zeros()
PermFx([1, 0, 0, 0]) | circuit
print(PermFx.pargs)
amplitude = Amplitude.run(circuit)
print(amplitude)
