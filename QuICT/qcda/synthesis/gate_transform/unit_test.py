#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 1:11 下午
# @Author  : Han Yu
# @File    : unit_test


import pytest

from .instruction_set import InstructionSet
from QuICT.core import *
from QuICT.qcda.synthesis import GateTransform

def test_gate_transform():
    for i in range(2, 3):
        circuit = Circuit(i)
        X | circuit
        gateSet = GateTransform(circuit)
        B = GateSet(circuit, with_copy=False)
        assert gateSet.equal(B)

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
