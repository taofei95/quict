#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 6:33 下午
# @Author  : Han Yu
# @File    : MCT_Linear_Simulation.py

from .._synthesis import Synthesis
from QuICT.models import *

def solve(n, m):
    circuit = Circuit(n)
    if m == 1:
        CX  | circuit([0, n - 1])
    elif m == 2:
        CCX | circuit([0, 1, n - 1])
    else:
        for i in range(m, 2, -1):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])
        CCX | circuit([0, 1, n - m + 1])
        for i in range(3, m + 1):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])

        for i in range(m - 1, 2, -1):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])
        CCX | circuit([0, 1, n - m + 1])
        for i in range(3, m):
            CCX | circuit([i - 1, n - 1 - (m - i + 1), n - 1 - (m - i)])
    return circuit

class MCT_Linear_Simulation_model(Synthesis):
    """
    Linear Simulation
    https://arxiv.org/abs/quant-ph/9503016
    Lemma 7.2
    """

    def __call__(self, m):
        self.pargs = [m]
        return self

    def build_gate(self):
        n = self.targets
        m = self.pargs[0]
        if m > (n // 2) + (1 if n % 2 == 1 else 0):
            raise Exception("控制位不能超过ceil(n/2)")
        if m < 1:
            raise Exception("至少要有一个控制位")
        return solve(self.targets, m)

MCT_Linear_Simulation = MCT_Linear_Simulation_model()
