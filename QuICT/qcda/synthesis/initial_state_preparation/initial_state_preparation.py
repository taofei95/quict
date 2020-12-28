#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/26 2:51
# @Author  : Han Yu
# @File    : _initial_state_preparation.py

import ctypes
from ctypes import c_int

import numpy as np
import os

from .._synthesis import Synthesis
from QuICT.core import *

# the allowed eps
EPS = 1e-13

def uniformlyRy(low, high, y):
    """ synthesis uniformRy gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        y(list<int>): the list of angle y
    Returns:
        the synthesis result
    """
    if low + 1 == high:
        GateBuilder.setGateType(GATE_ID["Ry"])
        GateBuilder.setTargs(low)
        GateBuilder.setPargs(y[0])
        return [GateBuilder.getGate()]
    length = len(y) // 2
    GateBuilder.setGateType(GATE_ID["CX"])
    GateBuilder.setTargs(high - 1)
    GateBuilder.setCargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((y[i] + y[i + length]) / 2)
        Rxn.append((y[i] - y[i + length]) / 2)
    gates = uniformlyRy(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyRy(low + 1, high, Rxn))
    gates.append(gateB)
    return gates

def uniformlyRz(low, high, z):
    """ synthesis uniformRz gate, bits range [low, high)
    Args:
        low(int): the left range low
        high(int): the right range high
        z(list<int>): the list of angle y
    Returns:
        the synthesis result
    """
    if low + 1 == high:
        GateBuilder.setGateType(GATE_ID["Rz"])
        GateBuilder.setTargs(low)
        print("?", z)
        GateBuilder.setPargs(z[1] - z[0])
        return [GateBuilder.getGate()]
    length = len(z) // 2
    GateBuilder.setGateType(GATE_ID["CX"])
    GateBuilder.setTargs(high - 1)
    GateBuilder.setCargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((z[i] + z[i + length]) / 2)
        Rxn.append((z[i] - z[i + length]) / 2)
    gates = uniformlyRz(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyRz(low + 1, high, Rxn))
    gates.append(gateB)
    return gates

class InitialStatePreparationOracle(Synthesis):
    """ initial state preparation

    """

    @property
    def initial_state_preparation_cdll(self):
        """

        Returns:
            _DLLT: the C++ library
        """
        if self.__initial_state_preparation_cdll is None:
            # sys = platform.system()
            path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "initial_state_preparation_cdll.so"
            self.__initial_state_preparation_cdll = ctypes.cdll.LoadLibrary(path)
        return self.__initial_state_preparation_cdll

    def __init__(self):
        """ initial function

        """
        super().__init__()
        self.__initial_state_preparation_cdll = None

    def __call__(self, other):
        """ add parameters with "()"

        Args:
            other: the parameters to add in, it can have follow forms:
                1) int/float/complex
                2) list<int/float/complex>
                3) tuple<int/float/complex>
        Raises:
            TypeException: the parameters filled in are wrong
        Returns:
            the initial_state_preparation_oracle filled by parameters.
        """

        if self.permit_element(other):
            self.pargs = [other]
        elif isinstance(other, list):
            self.pargs = []
            for element in other:
                if not self.permit_element(element):
                    raise TypeException("int or float or complex", element)
                self.pargs.append(element)
        elif isinstance(other, tuple):
            self.pargs = []
            for element in other:
                if not self.permit_element(element):
                    raise TypeException("int or float or complex", element)
                self.pargs.append(element)
        else:
            raise TypeException("int/float/complex or list<int/float/complex> or tuple<int/float/complex>", other)
        return self

    def build_gate(self):
        """ overload the function

        Returns:
            list<BasicGate>: the result

        """

        N = len(self.pargs)
        n = int(np.ceil(np.log2(N)))
        NN = 1 << n
        if NN > N:
            self.pargs.extend([0] * (NN - N))

        phases = []

        norm = 0
        for value in self.pargs:
            norm += abs(value)
        if abs(norm - 1) > 1e-10:
            for i in range(NN):
                self.pargs[i] /= norm
                phases.append(self.pargs[i])
                self.pargs[i] = abs(self.pargs[i])
                if self.pargs[i] > 0:
                    phases[i] /= self.pargs[i]
        else:
            for i in range(NN):
                phases.append(self.pargs[i])
                self.pargs[i] = abs(self.pargs[i])
                if self.pargs[i] > 0:
                    phases[i] /= self.pargs[i]

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
            raise Exception("the sum of input vector is 0")
        gates = []
        now = 0
        for i in range(n):
            add = (1 << i)
            alpha = back[now:now + add]
            flag = True
            for angle in alpha:
               test = np.floor(angle / np.pi)
               if abs(test * np.pi - angle) > 1e-13:
                   flag = False
                   break
            if not flag:
                gates.extend(uniformlyRy(0, i + 1, alpha))
            now += add
        # print(phases)
        # phases = [np.angle(phase) for phase in phases]
        # print(phases)
        # gates.extend(uniformlyRz(0, n, phases))
        return gates

InitialStatePreparation = InitialStatePreparationOracle()
