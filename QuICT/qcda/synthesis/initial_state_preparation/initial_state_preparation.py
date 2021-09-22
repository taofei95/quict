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
from QuICT.qcda.synthesis import uniformlyRy, uniformlyUnitary

# the allowed eps
EPS = 1e-13

__initial_state_preparation_cdll = None

def _initial_state_preparation_cdll():
    """

    Returns:
        _DLLT: the C++ library
    """
    global __initial_state_preparation_cdll
    if __initial_state_preparation_cdll is None:
        # sys = platform.system()
        path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "initial_state_preparation_cdll.so"
        __initial_state_preparation_cdll = ctypes.cdll.LoadLibrary(path)
    return __initial_state_preparation_cdll

def permit_element(element):
    """ judge whether the type of a parameter is int/float/complex

    for a quantum gate, the parameter should be int/float/complex

    Args:
        element: the element to be judged

    Returns:
        bool: True if the type of element is int/float/complex
    """
    if isinstance(element, int) or isinstance(element, float) or isinstance(element, complex):
        return True
    else:
        tp = type(element)
        if tp == np.int64 or tp == np.float or tp == np.complex128:
            return True
        return False

def InitialStatePreparationDecomposition(other):
    """
    Args:
        other: the parameters to add in, it can have follow forms:
            1) int/float/complex
            2) list<int/float/complex>
            3) tuple<int/float/complex>
    Raises:
        TypeException: the parameters filled in are wrong
    Returns:
        gateSet
    """

    pargs = [other]
    if permit_element(other):
        pargs = [other]
    elif isinstance(other, list):
        pargs = []
        for element in other:
            if not permit_element(element):
                raise TypeException("int or float or complex", element)
            pargs.append(element)
    elif isinstance(other, tuple):
        pargs = []
        for element in other:
            if not permit_element(element):
                raise TypeException("int or float or complex", element)
            pargs.append(element)
    else:
        raise TypeException("int/float/complex or list<int/float/complex> or tuple<int/float/complex>", other)

    N = len(pargs)
    n = int(np.ceil(np.log2(N)))
    NN = 1 << n
    if NN > N:
        pargs.extend([0] * (NN - N))

    phases = []

    norm = 0
    for value in pargs:
        norm += abs(value) * abs(value)
    if abs(norm - 1) > 1e-10:
        for i in range(NN):
            pargs[i] /= norm
            phases.append(np.angle(pargs[i]))
            pargs[i] = abs(pargs[i])
    else:
        for i in range(NN):
            phases.append(np.angle(pargs[i]))
            pargs[i] = abs(pargs[i])

    dll = _initial_state_preparation_cdll()
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
        np.array(pargs, dtype=np.longdouble),
        back
    )

    if safe == -1:
        raise Exception("the sum of input vector is 0")
    gates = CompositeGate()
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
            gates.extend(uniformlyRy(alpha))
        now += add
    unitaries = [np.diag([np.exp(1j * phases[2 * i]), np.exp(1j * phases[2 * i + 1])])
                 for i in range(len(phases) // 2)]
    # print(phases)
    # phases = [np.angle(phase) for phase in phases]
    gates.extend(uniformlyUnitary(unitaries))
    return gates

InitialStatePreparation = Synthesis(InitialStatePreparationDecomposition)
