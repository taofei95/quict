#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:42 下午
# @Author  : Han Yu
# @File    : SU2_rules.py

from .decomposition_rule import TransformRule

def _zyzRule(gate):
    pass
ZyzRule = TransformRule(_zyzRule)

def _zxzRule(gate):
    pass
ZxzRule = TransformRule(_zxzRule)

def _googleRule(gate):
    pass
GoogleRule = TransformRule(_googleRule)
