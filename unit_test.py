#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/2 9:48 上午
# @Author  : Dang Haoran
# @File    : unit_test

import pytest

#from QuICT.quda.synthesis.two_qubit_gate_rules import *
from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *


if __name__ == "__main__":
    #print(Crz2CxRule.check_equal())
    #print(Ryy2RxxRule.check_equal())
    #print(CZ2CXRule.check_equal())
    print(Rxx2CXRule.check_equal())
