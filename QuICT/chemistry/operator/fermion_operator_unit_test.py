#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/08/20 21:16
# @Author  : Xiaoquan Xu
# @File    : fermion_operator_unit_test.py

from QuICT.chemistry.operator.fermion_operator import FermionOperator
import pytest

def  test_construct():
    f_a = FermionOperator("1^ 4^ 13 2 405 2^ 5^ 5 ",-1.2)
    print(f_a)
    f_b = FermionOperator("  1^ 2^ 1 1^ 1",2)
    print(f_b)
    print(f_a+f_b)

    f_a = FermionOperator([(2,1),(8,0),(1,1),(2,0)],4008.2)
    f_b = FermionOperator([(2,0),(8,0),(1,1),(2,1)],-0.03)
    print(f_a)
    print(f_b)
    f_c = f_a + f_b
    print(f_c)

    f_c *= f_a
    print(f_c)
    f_c -= f_b
    print(f_c)
    f_c /= 4
    print(f_c)
    print(f_c.parse())
    assert 1

if __name__ == "__main__":
    pytest.main(["./fermion_operator_unit_test.py"])
