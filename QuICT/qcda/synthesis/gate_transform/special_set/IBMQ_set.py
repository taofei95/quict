#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 5:00 下午
# @Author  : Han Yu
# @File    : IBMQ_set.py

from .. import InstructionSet
from ..transform_rule import TransformRule

from QuICT.core import *

def _IBMQ_SU2(gate):
    pass

def _IBMQ_SU4(gate):
    pass

IBMQSet = InstructionSet(CX, [Rz, SX, X])
IBMQSet.register_SU2_rule(TransformRule(_IBMQ_SU2))
IBMQSet.register_SU4_rule(TransformRule(_IBMQ_SU4))
