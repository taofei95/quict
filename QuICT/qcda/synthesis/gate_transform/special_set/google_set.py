#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 5:00 下午
# @Author  : Han Yu
# @File    : IBMQ_set.py

from .. import InstructionSet
from ..transform_rule import TransformRule

from QuICT.core import *

def _Google_SU2():
    pass

GoogleSet = InstructionSet(FSim, [SX, SY, SW])
GoogleSet.register_SU2_rule(TransformRule(_Google_SU2))
