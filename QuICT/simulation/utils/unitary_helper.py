# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/27 2:02 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : unitary_helper.py
from functools import lru_cache

from QuICT.core.gate import Unitary


# tool function
def build_unitary_gate(compositeGate, unitary, targets: list):
    if not isinstance(targets, list):
        targets = [targets]

    targets.sort()
    ugate = Unitary(unitary).copy()
    ugate.targs = targets
    compositeGate.append(ugate)


# cost function
@lru_cache(1000)
def tensor_cost(a, b):
    return 1.0 * ((1 << b) ** 2 - (1 << a) ** 2)


@lru_cache(1000)
def tensor_both_cost(a, b):
    return 1.0 * ((1 << (b + a)) ** 2)


@lru_cache(1000)
def multiply_cost(k):
    return 1.0 * ((1 << k) ** 3)


@lru_cache(1000)
def multiply_vector_cost(k):
    return 1.0 * ((1 << k) ** 2)


# tool class
class dp:
    def __init__(self, args, value=0):
        self.set = set(args)
        self.length = len(self.set)
        self.value = value

    def merge(self, other, value=0):
        return dp(self.set | other.set, value)

    def merge_value(self, other):
        k = len(self.set | other.set)
        if len(self.set & other.set) == 0:
            return tensor_both_cost(self.length, other.length)
        return tensor_cost(self.length, k) + tensor_cost(other.length, k) + multiply_cost(k)

    def amplitude_cost(self, width):
        return self.value + tensor_cost(self.length, width) + multiply_vector_cost(width)
