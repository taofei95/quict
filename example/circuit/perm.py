#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/12 11:20 上午
# @Author  : Han Yu
# @File    : Perm.py
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import CircuitSimulator

circuit = Circuit(4)
PermFx(2, [0]) | circuit([0, 1, 2])

simulator = CircuitSimulator()
amplitude = simulator.run(circuit)
print(amplitude)
