#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:42 下午
# @Author  : Han Yu
# @File    : SU4_rules.py

from .decomposition_rule import TransformRule

def _cnotRule(gate):
    pass
CnotRule = TransformRule(_cnotRule)

def _fsimRule(gate):
    pass
FsimRule = TransformRule(_fsimRule)

