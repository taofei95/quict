#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/26 12:25 下午
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

from QuICT.models import *
from QuICT.optimization import cnot_ancillae

def test_1():
    for n in range(4, 5):
        circuit = Circuit(n)
        for _ in range(50):
            for i in range(n - 1):
                CX | circuit([i, i + 1])

def test_2():
    assert 1

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
