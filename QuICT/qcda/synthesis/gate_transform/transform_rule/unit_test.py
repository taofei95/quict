#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/28 1:25 上午
# @Author  : Han Yu
# @File    : unit_test

import pytest

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *

def test_SU2():
    assert XyxRule.check_equal()
    assert IbmqRule.check_equal()
    assert ZyzRule.check_equal()

def test_rules_transform():
    gateList = [CX, CY, CZ, CH, CRz, Rxx, Ryy, Rzz, FSim]
    from ..instruction_set import _generate_default_rule
    for i in range(len(gateList)):
        for j in range(len(gateList)):
            if i != j:
                rule = _generate_default_rule(gateList[i], gateList[j])
                print(gateList[i])
                print(gateList[j])
                assert rule.check_equal()

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
