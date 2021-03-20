#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 5:00 下午
# @Author  : Han Yu
# @File    : IBMQ_set.py

from .. import InstructionSet

def Google_SU2():
    pass
def Google_SU4():
    pass
GoogleSet = InstructionSet()
GoogleSet.register_SU2_rule(Google_SU2)
GoogleSet.register_SU4_rule(Google_SU4)
