#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/08/22 20:30
# @Author  : Xiaoquan Xu
# @File    : qubit_operator_unit_test.py

from QuICT.chemistry.operator.qubit_operator import QubitOperator
import pytest

def test_construction():
    f_a = QubitOperator("X1 Y4 Z13 X2 Y1", -1.2)
    f_A = QubitOperator([(1,2), (4,2), (2,1), (13,3), (1,1)], 1.2)
    f_b = QubitOperator("Z1 Y4 Z13 X2", 1.2j)
    assert f_a == f_A and f_a + f_b == QubitOperator(0)

    try:
        f_e = QubitOperator([(1,0)])
    except Exception:
        pass
    else:
        assert 0

    f_a = QubitOperator("X1 X1 Y1 Z1 Y1 Z1 X1 Y1")
    assert f_a == -1j * QubitOperator("Z1")

def test_operation():
    f_a = QubitOperator([(2,1),(8,3),(1,1),(2,2)],0.5)
    f_b = QubitOperator([(2,3),(8,3),(1,1)],-0.8)
    assert f_a * f_b == QubitOperator([], -0.4j)
    assert f_a + f_b == f_a * (1.6j+1)

    f_a = QubitOperator('X0 Y1') + QubitOperator('Y0 Y1')
    f_b = QubitOperator('Y0 X1') + QubitOperator('Z1 Z0') + QubitOperator('Y1 Z0')
    f_c = f_a * f_b + QubitOperator('X1 X0') - QubitOperator('Y0 X1') - QubitOperator('Z0 Z1')
    assert f_c == QubitOperator('X0',1j) - QubitOperator('Y0',1j) - QubitOperator('Z1',1j)

if __name__ == "__main__":
    pytest.main(["./qubit_operator_unit_test.py"])
