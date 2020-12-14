#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/10 11:39
# @Author  : Han Yu
# @File    : _QState_computing.py

from ctypes import *
import random

import numpy as np

from QuICT.backends import systemCdll
from QuICT.core.exception import *

"""

This file is to define the computing of the QState

"""

def QState_merge(qState, other):
    if qState.id == other.id:
        return
    if len(set(qState.qureg).intersection(set(other.qureg))) != 0:
        return

    dll = systemCdll.quick_operator_cdll
    merge_operator_func = dll.merge_operator_func
    merge_operator_func.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
    ]
    length = (1 << len(qState.qureg)) * (1 << len(other.qureg))
    merge_operator_func.restype = None
    values = np.zeros(length, dtype=np.complex)
    merge_operator_func(
        len(qState.qureg),
        qState.values,
        len(other.qureg),
        other.values,
        values
    )
    for qubit in other.qureg:
        qubit.qState = qState
    qState.qureg.extend(other.qureg)
    del other
    qState.values = values
    
def QState_deal_single_gate(qState, gate, has_fidelity = False, fidelity = 1.0):
    dll = systemCdll.quick_operator_cdll
    single_operator_func = dll.single_operator_func
    single_operator_func.argtypes = [
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
    ]

    index = 0
    qubit = qState.qureg.circuit.qubits[gate.targ]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        index = index + 1
    if index == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    matrix = gate.matrix
    if has_fidelity:
        theta = np.arccos(fidelity / np.sqrt(2)) - np.pi / 4
        theta *= (random.random() - 0.5) * 2
        RyMatrix = np.array(
            [
                np.cos(theta), -np.sin(theta),
                np.sin(theta), np.cos(theta)
            ]
            , dtype=np.complex
        )
        # print(fidelity, theta, RyMatrix)
        # matrix = RyMatrix * matrix
        Ry0 = RyMatrix[0] * matrix[0] + RyMatrix[1] * matrix[2]
        Ry1 = RyMatrix[0] * matrix[1] + RyMatrix[1] * matrix[3]
        Ry2 = RyMatrix[2] * matrix[0] + RyMatrix[3] * matrix[2]
        Ry3 = RyMatrix[2] * matrix[1] + RyMatrix[3] * matrix[3]
        matrix[0] = Ry0
        matrix[1] = Ry1
        matrix[2] = Ry2
        matrix[3] = Ry3
    single_operator_func(
        len(qState.qureg),
        index,
        qState.values,
        matrix
    )

def QState_deal_measure_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    measure_operator_func = dll.measure_operator_func
    measure_operator_func.argtypes = [
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        c_double,
        POINTER(c_double)
    ]
    measure_operator_func.restype = c_bool

    index = 0
    qubit = qState.qureg.circuit.qubits[gate.targ]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        index = index + 1
    if index == len(qState.qureg):
        raise FrameworkException("the index is out of range")
    generation = random.random()

    prob = c_double()
    result = measure_operator_func(
        len(qState.qureg),
        index,
        qState.values,
        generation,
        pointer(prob)
    )
    qState.qureg.remove(qubit)
    qState.values = qState.values[:(1 << len(qState.qureg))]
    qubit.qState = None
    qubit.measured = result
    qubit.prob = prob.value

def QState_deal_reset_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    reset_operator_func = dll.reset_operator_func
    reset_operator_func.argtypes = [
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
    ]

    index = 0
    qubit = qState.qureg.circuit.qubits[gate.targ]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        index = index + 1
    if index == len(qState.qureg):
        raise FrameworkException("the index is out of range")
    reset_operator_func(
        len(qState.qureg),
        index,
        qState.values
    )
    qState.qureg.remove(qubit)
    qState.values = qState.values[:(1 << len(qState.qureg))]
    qubit.qState = None

def QState_deal_control_single_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    control_single_operator_func = dll.control_single_operator_func
    control_single_operator_func.argtypes = [
        c_int,
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
    ]
    cindex = 0
    qubit = qState.qureg.circuit.qubits[gate.carg]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        cindex = cindex + 1
    if cindex == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    tindex = 0
    qubit = qState.qureg.circuit.qubits[gate.targ]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        tindex = tindex + 1
    if tindex == len(qState.qureg):
        raise FrameworkException("the index is out of range")
    control_single_operator_func(
        len(qState.qureg),
        cindex,
        tindex,
        qState.values,
        gate.matrix
    )

def QState_deal_ccx_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    ccx_single_operator_func = dll.ccx_single_operator_func
    ccx_single_operator_func.argtypes = [
        c_int,
        c_int,
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
    ]
    cindex1 = 0
    qubit = qState.qureg.circuit.qubits[gate.cargs[0]]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        cindex1 = cindex1 + 1
    if cindex1 == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    cindex2 = 0
    qubit = qState.qureg.circuit.qubits[gate.cargs[1]]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        cindex2 = cindex2 + 1
    if cindex2 == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    tindex = 0
    qubit = qState.qureg.circuit.qubits[gate.targ]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        tindex = tindex + 1
    if tindex == len(qState.qureg):
        raise FrameworkException("the index is out of range")
    ccx_single_operator_func(
        len(qState.qureg),
        cindex1,
        cindex2,
        tindex,
        qState.values
    )

def QState_deal_swap_gate(qState, gate):
    cindex = 0
    qubit = qState.qureg.circuit.qubits[gate.targs[0]]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        cindex = cindex + 1
    if cindex == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    tindex = 0
    qubit = qState.qureg.circuit.qubits[gate.targs[1]]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        tindex = tindex + 1
    if tindex == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    t = qState.qureg[cindex]
    qState.qureg[cindex] = qState.qureg[tindex]
    qState.qureg[tindex] = t

def QState_deal_custom_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    custom_operator_gate = dll.custom_operator_gate
    custom_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        POINTER(c_int),
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
    ]

    index = np.array([])
    for idx in gate.targs:
        qubit = qState.qureg.circuit.qubits[idx]
        temp_idx = 0
        for test in qState.qureg:
            if test.id == qubit.id:
                break
            temp_idx = temp_idx + 1
        if temp_idx == len(qState.qureg):
            raise FrameworkException("the index is out of range")
        np.append(index, temp_idx)

    custom_operator_gate(
        len(qState.qureg),
        qState.values,
        index,
        gate.targets,
        gate.matrix
    )

def QState_deal_perm_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    perm_operator_gate = dll.perm_operator_gate
    perm_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS")
    ]

    index = np.array([], dtype=np.int)
    targs = gate.targs
    if not isinstance(targs, list):
        targs = [targs]
    for idx in targs:
        qubit = qState.qureg.circuit.qubits[idx]
        temp_idx = 0
        for test in qState.qureg:
            if test.id == qubit.id:
                break
            temp_idx = temp_idx + 1
        if temp_idx == len(qState.qureg):
            raise FrameworkException("the index is out of range")
        index = np.append(index, temp_idx)
    perm_operator_gate(
        len(qState.qureg),
        qState.values,
        index,
        gate.targets,
        np.array(gate.pargs, dtype=np.int)
    )

def QState_deal_controlMulPerm_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    perm_operator_gate = dll.control_mul_perm_operator_gate
    perm_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]

    index = np.array([], dtype=np.int)
    targs = gate.targs
    if not isinstance(targs, list):
        targs = [targs]
    for idx in targs:
        qubit = qState.qureg.circuit.qubits[idx]
        temp_idx = 0
        for test in qState.qureg:
            if test.id == qubit.id:
                break
            temp_idx = temp_idx + 1
        if temp_idx == len(qState.qureg):
            raise FrameworkException("the index is out of range")
        index = np.append(index, temp_idx)
    control = gate.cargs[0]
    qubit = qState.qureg.circuit.qubits[control]
    temp_idx = 0
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        temp_idx = temp_idx + 1
    if temp_idx == len(qState.qureg):
        raise FrameworkException("the index is out of range")
    control = temp_idx
    perm_operator_gate(
        len(qState.qureg),
        qState.values,
        index,
        control,
        gate.targets,
        gate.pargs[0],
        gate.pargs[1]
    )

def QState_deal_shorInitial_gate(qState, gate):
    dll = systemCdll.quick_operator_cdll
    perm_operator_gate = dll.shor_classical_initial_gate
    perm_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        c_int,
        c_int
    ]

    index = np.array([], dtype=np.int)
    targs = gate.targs
    if not isinstance(targs, list):
        targs = [targs]
    for idx in targs:
        qubit = qState.qureg.circuit.qubits[idx]
        temp_idx = 0
        for test in qState.qureg:
            if test.id == qubit.id:
                break
            temp_idx = temp_idx + 1
        if temp_idx == len(qState.qureg):
            raise FrameworkException("the index is out of range")
        index = np.append(index, temp_idx)
    perm_operator_gate(
        len(qState.qureg),
        qState.values,
        index,
        gate.targets,
        gate.pargs[0],
        gate.pargs[1],
        gate.pargs[2]
    )
