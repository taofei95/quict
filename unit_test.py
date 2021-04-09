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
    # print(Cx2ChRule.check_equal()) 
    # print(Cx2CrzRule.check_equal()) 
    # print(Cx2RxxRule.check_equal())
    # print(Cx2RyyRule.check_equal()) 
    # print(Cx2RzzRule.check_equal()) 

    # Cy -> others

    # print(Cy2CxRule.check_equal())
    # print(Cy2CzRule.check_equal())
    # print(Cy2ChRule.check_equal())
    # print(Cy2CrzRule.check_equal())
    # print(Cy2RxxRule.check_equal())
    # print(Cy2RyyRule.check_equal())
    # print(Cy2RzzRule.check_equal()) 

    # Cz -> others

    # print(Cz2CxRule.check_equal())
    # print(Cz2CyRule.check_equal())
    # print(Cz2ChRule.check_equal())
    # print(Cz2CrzRule.check_equal())
    # print(Cz2RxxRule.check_equal())
    # print(Cz2RyyRule.check_equal()) 
    # print(Cz2RzzRule.check_equal()) 

    # Ch -> others
    # print(Ch2CxRule.check_equal())
    # print(Ch2CyRule.check_equal())
    # print(Ch2CzRule.check_equal())
    # print(Ch2CrzRule.check_equal())
    # print(Ch2RxxRule.check_equal())
    # print(Ch2RyyRule.check_equal()) 
    # print(Ch2RzzRule.check_equal())

    # Crz -> others

    # print(Crz2CxRule.check_equal())
    # print(Crz2CyRule.check_equal())
    # print(Crz2CzRule.check_equal())
    # print(Crz2ChRule.check_equal())
    # print(Crz2RxxRule.check_equal())
    # print(Crz2RyyRule.check_equal()) 
    # print(Crz2RzzRule.check_equal()) 

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
    print(Fsim2CRzRule.check_equal())

    print("Finish\n")
