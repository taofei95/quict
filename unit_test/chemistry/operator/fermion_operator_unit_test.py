#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/08/20 21:16
# @Author  : Xiaoquan Xu
# @File    : fermion_operator_unit_test.py

from QuICT.chemistry.operator.fermion_operator import FermionOperator
import pytest


def test_construction():
    f_a = FermionOperator("1^ 4^ 3 2 2^ 5 4 2 5^ 2 4^", -1.2)
    assert f_a == FermionOperator()
    f_a = FermionOperator("1^ 4^ 3 2 2^ 5 4 2 5^ 2^ 4^", -1.2)
    f_A = FermionOperator([(4, 1), (3, 0), (2, 0), (5, 0), (1, 1), (2, 1), (5, 1)], 1.2)
    assert f_a == f_A

    f_b = FermionOperator(" 1^ 4^ 2 3 2^ ", 2.)
    f_B = FermionOperator([(1, 1), (4, 1), (3, 0)], -2) + FermionOperator([(2, 1), (1, 1), (3, 0), (2, 0), (4, 1)], -2)
    assert f_b == f_B

    f_c = FermionOperator([(1, 1), (2, 0), (2, 1), (3, 0), (4, 1)], 3.2)
    f_d = FermionOperator([(1, 1), (2, 0), (3, 0), (4, 1), (5, 1), (2, 1), (5, 0)], 1.2)
    assert f_a + f_b == f_c + f_d


def test_operation():
    f_a = FermionOperator([(2, 1), (8, 0), (1, 1), (2, 0)], -0.2)
    f_b = FermionOperator([(2, 0), (8, 0), (1, 1), (2, 1)], -0.4)
    assert f_a - f_b == f_a + (-1) * f_b

    f_c = f_a + f_b
    f_C = FermionOperator([(1, 1), (2, 1), (2, 0), (8, 0)], -0.2) + FermionOperator([(1, 1), (8, 0)], 0.4)
    assert f_c == f_C

    f_c = FermionOperator([(2, 0), (3, 1), (1, 0)], 0.5)
    assert f_a * f_c == FermionOperator()
    assert f_c * f_a == FermionOperator('1^ 1 2 3^ 8', -0.1) + FermionOperator('2 3^ 8', 0.1)

    f_c = FermionOperator([(2, 1), (8, 1), (1, 0)], 5)
    assert f_a * f_b == FermionOperator()
    assert f_a * f_c * f_b == FermionOperator('1^ 8 2^', 0.4)


if __name__ == "__main__":
    pytest.main(["./fermion_operator_unit_test.py"])
