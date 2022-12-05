#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/8 3:29 下午
# @Author  : Han Yu
# @File    : Rzz.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import H, Rzz
from QuICT.simulation.state_vector import CircuitSimulator


circuit = Circuit(3)
H | circuit
Rzz(np.pi / 2) | circuit([0, 1])

simulator = CircuitSimulator()
amplitude = simulator.run(circuit)
print(amplitude)