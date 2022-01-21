#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/16 10:04 下午
# @Author  : Han Yu, Kaiqi Li
# @File    : CCX_Dec.py
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


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

simulator = ConstantStateVectorSimulator(
    precision="double",
    optimize=False,
    gpu_device_id=0,
    sync=True
)

amplitude = simulator.run(circuit=circuit)
print(amplitude)
