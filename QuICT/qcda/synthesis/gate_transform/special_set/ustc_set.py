#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 4:41 下午
# @Author  : Han Yu
# @File    : ustc_set

from .. import InstructionSet
from ..transform_rule import TransformRule, ZyzRule

from QuICT.core import *

USTCSet = InstructionSet(CX, [Rx, Ry, Rz, H, X])
USTCSet.register_SU2_rule(ZyzRule)
