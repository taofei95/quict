#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/13 10:07 下午
# @Author  : Han Yu
# @File    : gate_transform

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_transform import GateTransform

if __name__ == "__main__":
    circuit = Circuit(5)
    # circuit.random_append(10)
    CH | circuit([0, 1])
    CY | circuit([1, 2])
    CZ | circuit([2, 3])
    CRz | circuit([3, 4])
    circuit.draw()
    GT = GateTransform()
    compositeGate = GT.execute(circuit)

    new_circuit = Circuit(5)
    new_circuit.extend(compositeGate.gates)
    new_circuit.draw()
