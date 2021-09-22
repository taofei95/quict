#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 3:32 下午
# @Author  : Han Yu
# @File    : ionq_set

from .. import InstructionSet
from ..transform_rule import XyxRule

from QuICT.core import *

IonQSet = InstructionSet([Rxx, Rx, Ry, Rz])
IonQSet.register_SU2_rule(XyxRule)
