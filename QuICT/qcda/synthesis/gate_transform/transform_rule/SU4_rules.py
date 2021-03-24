#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:42 下午
# @Author  : Han Yu
# @File    : SU4_rules.py

"""

the file describe TransformRule the decomposite SU(4) into instruction set.

"""

from .transform_rule import TransformRule

def _cnotRule(gate):
    pass
CnotRule = TransformRule(_cnotRule)

def _fsimRule(gate):
    pass
FsimRule = TransformRule(_fsimRule)

