#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 5:00 下午
# @Author  : Han Yu
# @File    : IBMQ_set.py

from .. import InstructionSet

def IBMQ_SU2():
    pass
def IBMQ_SU4():
    pass
IBMQSet = InstructionSet()
IBMQSet.register_SU2_rule(IBMQ_SU2)
IBMQSet.register_SU4_rule(IBMQ_SU4)
