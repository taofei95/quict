#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/26 2:51 下午
# @Author  : Han Yu
# @File    : _initial_state_preparation.py

from QuICT.synthesis._synthesis import Synthesis
from QuICT.models import *
import numpy as np
import os
import ctypes
from ctypes import c_int

EPS = 1e-13

def uniformlyRy(low, high, y):
    if low + 1 == high:
        GateBuilder.setGateType(GateType.Ry)
        GateBuilder.setTargs(low)
        GateBuilder.setPargs(y[0])
        return [GateBuilder.getGate()]
    length = len(y) // 2
    GateBuilder.setGateType(GateType.CX)
    GateBuilder.setTargs(high - 1)
    GateBuilder.setCargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((y[i] + y[i + length]) / 2)
        Rxn.append((y[i] - y[i + length]) / 2)
    del y
    gates = uniformlyRy(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyRy(low + 1, high, Rxn))
    gates.append(gateB)
    return gates

class initial_state_preparation_oracle(Synthesis):
    """
    类的属性
    """

    @property
    def initial_state_preparation_cdll(self):
        """
        :return: 懒加载alpha计算库
        """
        if self.__initial_state_preparation_cdll is None:
            # sys = platform.system()
            path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "initial_state_preparation_cdll.so"
            self.__initial_state_preparation_cdll = ctypes.cdll.LoadLibrary(path)
        return self.__initial_state_preparation_cdll

    def __init__(self):
        super().__init__()
        self.__initial_state_preparation_cdll = None

    def build_gate(self):
        N = len(self.pargs)
        n = int(np.ceil(np.log2(N)))
        NN = 1 << n

        dll = self.initial_state_preparation_cdll
        state_theta_computation = dll.state_theta_computation
        state_theta_computation.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.longdouble, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.longdouble, ndim=1, flags="C_CONTIGUOUS"),
        ]
        state_theta_computation.restype = c_int

        back = np.zeros(NN - 1, dtype=np.longdouble)
        safe = state_theta_computation(
            N,
            np.array(self.pargs, dtype=np.longdouble),
            back
        )

        if safe == -1:
            raise Exception("输入的向量不是单位向量")
        gates = []
        now = 0
        for i in range(n):
            add = (1 << i)
            alpha = back[now:now + add]
            print(alpha)
            gates.extend(uniformlyRy(0, i + 1, alpha))
            now += add
        return gates

initial_state_preparation = initial_state_preparation_oracle()
