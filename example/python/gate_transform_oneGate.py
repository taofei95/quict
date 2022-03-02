#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/3 7:19 下午
# @Author  : Han Yu
# @File    : gate_transform_oneGate

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_transform import *

if __name__ == "__main__":
    circuit = Circuit(2)
    CX | circuit
    circuit.draw()
    compositeGate = GateTransform.execute(circuit, GoogleSet)
    circuit.extend(compositeGate.gates)
    circuit.draw()
