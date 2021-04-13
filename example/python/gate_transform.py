#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/13 10:07 下午
# @Author  : Han Yu
# @File    : gate_transform

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *

if __name__ == "__main__":
    circuit = Circuit(5)
    circuit.random_append(10)
    circuit.draw_photo(show_depth=False)
    compositeGate = GateTransform(circuit, USTCSet)

    new_circuit = Circuit(5)
    new_circuit.set_exec_gates(compositeGate)
    new_circuit.draw_photo(show_depth=False)
