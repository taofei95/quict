#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/13 10:07 下午
# @Author  : Han Yu
# @File    : gate_transform

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.virtual_machine.special_set import USTCSet
from QuICT.qcda.synthesis.gate_transform import GateTransform


if __name__ == "__main__":
    circuit = Circuit(5)
    CH | circuit([0, 1])
    CY | circuit([1, 2])
    CZ | circuit([2, 3])
    CRz | circuit([3, 4])
    circuit.draw(method="command")

    GT = GateTransform(instruction_set=USTCSet)
    new_circuit = GT.execute(circuit)
    new_circuit.draw(method="command") #, flatten=True)
