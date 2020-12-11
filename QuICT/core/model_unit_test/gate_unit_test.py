#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/23 4:30 下午
# @Author  : Han Yu
# @File    : model_unit_test.py

import pytest

import numpy as np

from QuICT import *
from QuICT.algorithm import SyntheticalUnitary

def test_permMulDetail():
    max_test = 5
    every_round = 20
    for i in range(4, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ControlPermMulDetail([2, 5]) | circuit
            ControlPermMulDetail([2, 5]).inverse() | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    pytest.main(["./gate_unit_test.py"])
