#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 4:41 下午
# @Author  : Han Yu
# @File    : ustc_set

from .. import InstructionSet
from ..transform_rule import TransformRule, ZyzRule

from QuICT.core import *

def _IonQ_SU4(gate):
    pass

USTCSet = InstructionSet(CX, [Rx, Ry, Rz, H, X])
USTCSet.register_SU2_rule(TransformRule(ZyzRule))
USTCSet.register_SU4_rule(TransformRule(_IonQ_SU4))
