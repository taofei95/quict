#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/28 1:25 上午
# @Author  : Han Yu
# @File    : unit_test

import pytest

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *

if __name__ == "__main__":
    # print(Crz2CxRule.check_equal())
    # print(Cx2CyRule.check_equal())
    # print(Cx2CzRule.check_equal())
    # print(Cx2ChRule.check_equal())
    # X print(Cx2CrzRule.check_equal())  print(gateSet.matrix())
    # print(Cy2CzRule.check_equal())
    # print(Cy2ChRule.check_equal())
    # print(Cz2CyRule.check_equal())
    # print(Cz2ChRule.check_equal())
    # X print(Cz2CrzRule.check_equal())
    # print(Crz2CyRule.check_equal())
    # print(Crz2CzRule.check_equal())
    # print(Crz2ChRule.check_equal())
    # print(Ch2CyRule.check_equal())
    # print(Ch2CzRule.check_equal())
    print(Rxx2CxRule.check_equal())
    print("16\n")
