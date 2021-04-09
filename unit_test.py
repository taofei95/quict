#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/2 9:48 上午
# @Author  : Dang Haoran
# @File    : unit_test

import pytest

from QuICT.core import *
from QuICT.qcda.synthesis.gate_transform import *

if __name__ == "__main__":
    # Cx -> others
    
    # print(Cx2CyRule.check_equal())
    # print(Cx2CzRule.check_equal())
    # print(CX2CHRule.check_equal()) 
    # print(CX2CRzRule.check_equal()) 
    # print(CX2RxxRule.check_equal())
    # print(CX2RyyRule.check_equal()) 
    # print(CX2RzzRule.check_equal()) 

    # CY -> others

    # print(CY2CXRule.check_equal())
    # print(CY2CZRule.check_equal())
    # print(CY2CHRule.check_equal())
    # print(CY2CRzRule.check_equal())
    # print(CY2RxxRule.check_equal())
    # print(CY2RyyRule.check_equal())
    # print(CY2RzzRule.check_equal()) 

    # CZ -> others

    # print(CZ2CXRule.check_equal())
    # print(CZ2CYRule.check_equal())
    # print(CZ2CHRule.check_equal())
    # print(CZ2CRzRule.check_equal())
    # print(CZ2RxxRule.check_equal())
    # print(CZ2RyyRule.check_equal()) 
    # print(CZ2RzzRule.check_equal()) 
    # print(Cz2FsimRule.check_equal())

    # CH -> others
    # print(CH2CXRule.check_equal())
    # print(CH2CYRule.check_equal())
    # print(CH2CZRule.check_equal())
    # print(CH2CRzRule.check_equal())
    # print(CH2RxxRule.check_equal())
    # print(CH2RyyRule.check_equal()) 
    # print(CH2RzzRule.check_equal())

    # CRz -> others

    # print(CRz2CXRule.check_equal())
    # print(CRz2CYRule.check_equal())
    # print(CRz2CZRule.check_equal())
    # print(CRz2CHRule.check_equal())
    # print(CRz2RxxRule.check_equal())
    # print(CRz2RyyRule.check_equal()) 
    # print(CRz2RzzRule.check_equal()) 

    # Rxx -> others

    # print(Rxx2CxRule.check_equal())
    # print(Rxx2CyRule.check_equal())
    # print(Rxx2CzRule.check_equal())
    # print(Rxx2ChRule.check_equal())
    # print(Rxx2CrzRule.check_equal())
    # print(Rxx2RyyRule.check_equal())
    # print(Rxx2RzzRule.check_equal())

    # Ryy -> others

    # print(Ryy2CxRule.check_equal())
    # print(Ryy2CyRule.check_equal())
    # print(Ryy2CzRule.check_equal())
    # print(Ryy2ChRule.check_equal())
    # print(Ryy2CrzRule.check_equal())
    # print(Ryy2RxxRule.check_equal())
    # print(Ryy2RzzRule.check_equal())

    # Rzz -> others

    # print(Rzz2CxRule.check_equal())
    # print(Rzz2CyRule.check_equal())
    # print(Rzz2CzRule.check_equal())
    # print(Rzz2ChRule.check_equal())
    # print(Rzz2CrzRule.check_equal())
    # print(Rzz2RxxRule.check_equal())
    # print(Rzz2RyyRule.check_equal())
    # print(ZyzRule.check_equal())

    # print(Cx2FsimRule.check_equal())
    # print(Cy2FsimRule.check_equal())
    # print(Ch2FsimRule.check_equal())

    # print(IbmqRule.check_equal())
    print(Fsim2CRzRule.check_equal())
    print("99\n")
