#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51 上午
# @Author  : Han Yu
# @File    : _unit_test.py

from QuICT.models import *
from QuICT.algorithm import *
from QuICT.synthesis.initial_state_preparation import initial_state_preparation

if __name__ == "__main__":
    circuit = Circuit(2)
    values = [0.5, 0, 0.25, 0.25]
    initial_state_preparation(values) | circuit
    circuit.print_infomation()
    print(Amplitude.run(circuit))

    i = 8
    circuit = Circuit(i)
    values = [1.0 / (1 << i) for _ in range(1 << i)]
    print(values)
    initial_state_preparation(values) | circuit
    circuit.print_infomation()
    print(Amplitude.run(circuit))
