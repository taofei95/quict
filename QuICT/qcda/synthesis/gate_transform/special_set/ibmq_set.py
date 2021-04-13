#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/4 10:01 上午
# @Author  : Han Yu
# @File    : ibmq_set

from .. import InstructionSet
from ..transform_rule import TransformRule, IbmqRule

from QuICT.core import *

IBMQSet = InstructionSet(CX, [Rz, SX, X])
IBMQSet.register_SU2_rule(IbmqRule)
