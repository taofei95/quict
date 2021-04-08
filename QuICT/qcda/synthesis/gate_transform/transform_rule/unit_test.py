#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/28 1:25 上午
# @Author  : Han Yu
# @File    : unit_test

import pytest

# from .two_qubit_gate_rules import *
from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *


if __name__ == "__main__":
    for i in range(100):
        print(ZyzRule.check_equal())
