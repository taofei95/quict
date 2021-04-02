#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:42 下午
# @Author  : Han Yu
# @File    : SU2_rules.py

"""

the file describe TransformRule the decomposite SU(2) into instruction set.

"""

from .transform_rule import TransformRule

class SU2TransformRule(TransformRule):
    def check_equal(self, ignore_phase = True, eps = 1e-13):
        pass

def _zyzRule(gate):
    pass
ZyzRule = TransformRule(_zyzRule)

def _xyxRule(gate):
    pass
ZxzRule = TransformRule(_zxzRule)

def _ibmqRule(gate):
    pass
IBMQRule = TransformRule(_ibmqRule)

def _googleRule(gate):
    pass
GoogleRule = TransformRule(_googleRule)
