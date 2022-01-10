#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/23 4:30
# @Author  : Han Yu
# @File    : model_unit_test.py
import pytest
import random

from QuICT.core import Circuit, Qureg
from QuICT.core.gate import *


def test_gate_build():
    cir = Circuit(10)
    # single qubit gate
    h1 = HGate()
    h1 | cir(1)         # 1
    H | cir             # 11
    
    # single qubit gate with param
    my_u1 = U1Gate([1])
    my_u1 | cir(2)      # 12
    U1(0) | cir(1)      # 13

    # two qubit gate
    my_CX = CX & [3, 4]
    my_CX | cir         # 14
    CX | cir([3, 4])    # 15

    # two qubit gate with param
    CU3(1, 0, 0) | cir([5, 6])  # 16
    
    # complexed gate
    CCRz(1) | cir([7, 8, 9])    # 17
    cg_ccrz = CCRz.build_gate()
    cg_ccrz | cir([9, 8, 7])    # 22

    assert len(cir.gates) == 22


def test_gate_name():
    my_gate = PhaseGate()
    assert len(my_gate.name.split('-')) == 1

    q = Qureg(1)
    g2 = my_gate & q
    assert len(g2.name.split('-')) == 2

    cir = Circuit(q)
    g2 | cir
    assert len(cir.gates[0].name.split('-')) == 3


if __name__ == "__main__":
    pytest.main(["./gate_unit_test.py"])
