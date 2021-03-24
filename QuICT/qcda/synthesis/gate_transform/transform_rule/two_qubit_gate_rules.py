#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:44 下午
# @Author  : Han Yu
# @File    : two_qubit_gate_rules.py

from .transform_rule import TransformRule
from QuICT.core import *

"""

the file describe TransformRule between two kinds of 2-qubit gates.

"""

def _cx2rxx_rule(gate):
    pass
Cx2RxxRule = TransformRule(_cx2rxx_rule, CX)
