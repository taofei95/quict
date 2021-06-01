#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 12:13
# @Author  : Han Yu
# @File    : _exec_operator.py

"""

This file is to define the execute operator
to be inherited by more than one gate.

"""

from ctypes import *
import random

import numpy as np

from QuICT.backends import systemCdll
from QuICT.core.exception import *

# gate exec operator

def exec_single(gate, circuit):
    """ apply an one-qubit gate on this circuit

    Args:
        gate(BasicGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState = circuit.qubits[gate.targ].qState
    QState_deal_single_gate(qState, gate, circuit.fidelity)

def exec_controlSingle(gate, circuit):
    """ apply a controlled one-qubit gate on this qState

    Args:
        gate(BasicGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState0 = circuit.qubits[gate.carg].qState
    qState1 = circuit.qubits[gate.targ].qState
    QState_merge(qState0, qState1)
    QState_deal_control_single_gate(qState0, gate)

def exec_two(gate, circuit):
    """ apply a two-qubit gate on this qState

    Args:
        gate(BasicGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState0 = circuit.qubits[gate.targs[0]].qState
    qState1 = circuit.qubits[gate.targs[1]].qState
    QState_merge(qState0, qState1)
    QState_deal_two_qubit_gate(qState0, gate)

def exec_toffoli(gate, circuit):
    """ apply a toffoli gate on this qState

    Args:
        gate(CCXGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState0 = circuit.qubits[gate.cargs[0]].qState
    qState1 = circuit.qubits[gate.cargs[1]].qState
    qState2 = circuit.qubits[gate.targ].qState
    QState_merge(qState0, qState1)
    QState_merge(qState0, qState2)
    QState_deal_ccx_gate(qState0, gate)

def exec_measure(gate, circuit):
    """ apply a measure gate on this qState

    Note that after flush the measure gate, the qubit will be removed
    from the qState.

    Args:
        gate(MeasureGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState0 = circuit.qubits[gate.targ].qState
    QState_deal_measure_gate(qState0, gate)

def exec_reset(gate, circuit):
    """ apply a reset gate on this qState

    Note that after flush the reset gate, the qubit will be removed
    from the qState.

    Args:
        gate(ResetGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState0 = circuit.qubits[gate.targ].qState
    QState_deal_reset_gate(qState0, gate)

def exec_barrier(gate, circuit):
    pass

def exec_swap(gate, circuit):
    """ apply a swap gate on this qState

    Args:
        gate(SwapGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState0 = circuit.qubits[gate.targs[0]].qState
    qState1 = circuit.qubits[gate.targs[1]].qState
    QState_merge(qState0, qState1)
    QState_deal_swap_gate(qState0, gate)

def exec_perm(gate, circuit):
    """ apply a Perm gate on this qState

    Args:
        gate(PermGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt.

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState = circuit.qubits[gate.targ].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        QState_merge(qState, new_qState)
    QState_deal_perm_gate(qState, gate)

def exec_unitary(gate, circuit):
    """ apply a unitary gate on this qState

    Args:
        gate(UnitaryGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt.

    Exceptions:
        FrameworkException: the index is out of range
    """
    qState = circuit.qubits[gate.targ].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        QState_merge(qState, new_qState)
    QState_deal_unitary_gate(qState, gate)

def exec_shorInit(gate, circuit):
    """ apply a shorInitial gate on this qState

    Args:
        gate(shorInitialGate): the gate to be applied.
        circuit(Circuit): the circuit to be dealt.

    Exceptions:
        FrameworkException: the index is out of range
    """

    qState = circuit.qubits[gate.targ].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        QState_merge(qState, new_qState)
    QState_deal_shorInitial_gate(qState, gate)

def exec_controlMulPerm(gate, circuit):
    """ apply a controlMulPerm gate on this qState

    Args:
        gate(controlMulPerm): the gate to be applied.
        circuit(Circuit): the circuit to be dealt.

    Exceptions:
        FrameworkException: the index is out of range
    """

    qState = circuit.qubits[gate.carg].qState
    for i in range(gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        QState_merge(qState, new_qState)
    QState_deal_controlMulPerm_gate(qState, gate)

# qState exec operator

def QState_merge(qState, other):
    """ merge another qState into this qState

    Args:
        qState(QState): the QState to be dealt
        other(QState): the qState need to be merged.

    Exceptions:
        FrameworkException: the index is out of range
    """
    if qState.id == other.id:
        return
    if len(set(qState.qureg).intersection(set(other.qureg))) != 0:
        return

    dll = systemCdll.quick_operator_cdll
    merge_operator_func = dll.merge_operator_func
    merge_operator_func.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    length = (1 << len(qState.qureg)) * (1 << len(other.qureg))
    merge_operator_func.restype = None
    values = np.zeros(length, dtype=np.complex64)
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

def QState_deal_single_gate(qState, gate, fidelity):
    """ apply an one-qubit gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(BasicGate): the gate to be applied.
        fidelity(float): the fidelity of the gate

    Exceptions:
        FrameworkException: the index is out of range
    """
    dll = systemCdll.quick_operator_cdll
    single_operator_func = dll.single_operator_func
    single_operator_func.argtypes = [
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
    ]

    index = 0
    qubit = qState.qureg.circuit.qubits[gate.targ]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        index = index + 1
    if index == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    matrix = gate.matrix.flatten()
    if fidelity is not None:
        theta = np.arccos(fidelity / np.sqrt(2)) - np.pi / 4
        theta *= (random.random() - 0.5) * 2
        RyMatrix = np.array(
            [
                np.cos(theta), -np.sin(theta),
                np.sin(theta), np.cos(theta)
            ]
            , dtype=np.complex64
        )
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

def QState_deal_two_qubit_gate(qState, gate):
    """ apply an two-qubit gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(BasicGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """

    dll = systemCdll.quick_operator_cdll
    two_qubit_operator_func = dll.two_qubit_operator_func
    two_qubit_operator_func.argtypes = [
        c_int,
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
    ]
    index1 = 0
    qubit = qState.qureg.circuit.qubits[gate.targs[0]]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        index1 = index1 + 1
    if index1 == len(qState.qureg):
        raise FrameworkException("the index is out of range")

    index2 = 0
    qubit = qState.qureg.circuit.qubits[gate.targs[1]]
    for test in qState.qureg:
        if test.id == qubit.id:
            break
        index2 = index2 + 1
    if index2 == len(qState.qureg) or index2 == index1:
        raise FrameworkException("the index is out of range")
    two_qubit_operator_func(
        len(qState.qureg),
        index1,
        index2,
        qState.values,
        gate.matrix.flatten()
    )

def QState_deal_control_single_gate(qState, gate):
    """ apply a controlled one-qubit gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(BasicGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """

    dll = systemCdll.quick_operator_cdll
    control_single_operator_func = dll.control_single_operator_func
    control_single_operator_func.argtypes = [
        c_int,
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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
    if tindex == len(qState.qureg) or tindex == cindex:
        raise FrameworkException("the index is out of range")
    control_single_operator_func(
        len(qState.qureg),
        cindex,
        tindex,
        qState.values,
        gate.matrix.flatten()
    )

def QState_deal_ccx_gate(qState, gate):
    """ apply a toffoli gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(BasicGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """
    dll = systemCdll.quick_operator_cdll
    ccx_single_operator_func = dll.ccx_single_operator_func
    ccx_single_operator_func.argtypes = [
        c_int,
        c_int,
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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

def QState_deal_measure_gate(qState, gate):
    """ apply a measure gate on this qState

    Note that after flush the measure gate, the qubit will be removed
    from the qState.

    Args:
        qState(QState): the QState to be dealt
        gate(MeasureGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """

    dll = systemCdll.quick_operator_cdll
    measure_operator_func = dll.measure_operator_func
    measure_operator_func.argtypes = [
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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
    """ apply a reset gate on this qState

    Note that after flush the reset gate, the qubit will be removed
    from the qState.

    Args:
        qState(QState): the QState to be dealt
        gate(ResetGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """
    dll = systemCdll.quick_operator_cdll
    reset_operator_func = dll.reset_operator_func
    reset_operator_func.argtypes = [
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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

def QState_deal_swap_gate(qState, gate):
    """ apply a swap gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(SwapGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """
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

def QState_deal_perm_gate(qState, gate):
    """ apply a Perm gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(PermGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """
    dll = systemCdll.quick_operator_cdll
    perm_operator_gate = dll.perm_operator_gate
    perm_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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

def QState_deal_unitary_gate(qState, gate):
    """ apply a custom gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(UnitaryGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """

    dll = systemCdll.quick_operator_cdll
    unitary_operator_gate = dll.unitary_operator_gate
    unitary_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
        POINTER(c_int),
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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

    unitary_operator_gate(
        len(qState.qureg),
        qState.values,
        index,
        gate.targets,
        gate.matrix.flatten()
    )

def QState_deal_shorInitial_gate(qState, gate):
    """ apply a shorInitial gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(shorInitialGate): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """

    dll = systemCdll.quick_operator_cdll
    perm_operator_gate = dll.shor_classical_initial_gate
    perm_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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

def QState_deal_controlMulPerm_gate(qState, gate):
    """ apply a controlMulPerm gate on this qState

    Args:
        qState(QState): the QState to be dealt
        gate(controlMulPerm): the gate to be applied.

    Exceptions:
        FrameworkException: the index is out of range
    """

    dll = systemCdll.quick_operator_cdll
    perm_operator_gate = dll.control_mul_perm_operator_gate
    perm_operator_gate.argtypes = [
        c_int,
        np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C_CONTIGUOUS"),
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
