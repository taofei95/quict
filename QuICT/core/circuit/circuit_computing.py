#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 10:31
# @Author  : Han Yu
# @File    : _circuit_computing.py

from ctypes import c_int
import random

import numpy as np

from QuICT.backends.systemcdll import systemCdll

def _getRandomList(l, n):
    """ get l number from 0, 1, ..., n - 1 randomly.
    Args:
        l(int)
        n(int)
    Returns:
        list<int>: the list of l random numbers
    """
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:l]

def inner_partial_prob(circuit, indexes):
    """ calculate the probabilities of the measure result of partial qureg in circuit

    Note that the function "flush" will be called before calculating
    this function is a cheat function, which do not change the state of the qureg.

    Args:
        circuit(Circuit): the circuit to be dealt
        indexes(list<int>): the indexes of the partial qureg.

    Returns:
        list<float>: the probabilities of the measure result, the memory mode is LittleEndian.

    """
    circuit.exec()
    dll = systemCdll.quick_operator_cdll
    partial_prob_operator = dll.partial_prob_cheat_operator
    partial_prob_operator.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
        c_int,
        c_int,
        np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
    ]

    length = 1 << len(indexes)

    partial_prob_operator.restype = np.ctypeslib.ndpointer(dtype=np.float64, shape=(length,))

    tangle_list = []
    tangle_values = np.array([], dtype=np.complex128)
    tangle_length = np.array([], dtype=np.int64)
    qubit_map = np.array([i for i in range(len(indexes))], dtype=np.int64)

    tangle_iter = 0
    for index in indexes:
        qubit = circuit[index]
        if qubit.qState not in tangle_list:
            tangle_list.append(qubit.qState)
    for tangle in tangle_list:
        tangle_values = np.append(tangle_values, tangle.values)
        tangle_length = np.append(tangle_length, len(tangle.qureg))
        for i in range(len(indexes)):
            index = indexes[i]
            qubit = circuit[index]
            if qubit.qState == tangle:
                qubit_map[i] = tangle_iter + tangle.index_for_qubit(qubit)
        tangle_iter = tangle_iter + len(tangle.qureg)

    ndpointer = partial_prob_operator(
        tangle_values,
        tangle_length,
        len(tangle_list),
        len(indexes),
        qubit_map
    )
    values = np.ctypeslib.as_array(ndpointer, shape=(length,))
    return values.tolist()

def inner_random_append(circuit, rand_size=10, typeList=None):
    from QuICT.core import GateBuilder, GATE_ID
    if typeList is None:
        typeList = [GATE_ID["Rx"], GATE_ID["Ry"], GATE_ID["Rz"],
                    GATE_ID["CX"], GATE_ID["CY"], GATE_ID["CRz"], GATE_ID["CH"], GATE_ID["CZ"],
                    GATE_ID["Rxx"], GATE_ID["Ryy"], GATE_ID["Rzz"], GATE_ID["FSim"]
                    ]
    qubit = circuit.circuit_width()
    for _ in range(rand_size):
        rand_type = random.randrange(0, len(typeList))
        GateBuilder.setGateType(typeList[rand_type])

        targs = GateBuilder.getTargsNumber()
        cargs = GateBuilder.getCargsNumber()
        pargs = GateBuilder.getParamsNumber()

        tclist = _getRandomList(targs + cargs, qubit)
        if targs != 0:
            GateBuilder.setTargs(tclist[:targs])
        if cargs != 0:
            GateBuilder.setCargs(tclist[targs:])
        if pargs != 0:
            params = []
            for _ in range(pargs):
                params.append(random.uniform(0, 2 * np.pi))
            GateBuilder.setPargs(params)
        gate = GateBuilder.getGate()
        circuit.gates.append(gate)

def inner_matrix_product_to_circuit(circuit, gate) -> np.ndarray:
    q_len = len(circuit.qubits)
    n = 1 << len(circuit.qubits)

    new_values = np.zeros((n, n), dtype=np.complex128)
    targs = gate.targs
    cargs = gate.cargs
    if not isinstance(targs, list):
        targs = [targs]
    if not isinstance(cargs, list):
        cargs = [cargs]
    targs = np.append(np.array(cargs, dtype=int).ravel(), np.array(targs, dtype=int).ravel())
    targs = targs.tolist()
    xor = (1 << q_len) - 1
    if not isinstance(targs, list):
        raise Exception("unknown error")
    matrix = gate.compute_matrix.reshape(1 << len(targs), 1 << len(targs))
    datas = np.zeros(n, dtype=int)
    for i in range(n):
        nowi = 0
        for kk in range(len(targs)):
            k = q_len - 1 - targs[kk]
            if (1 << k) & i != 0:
                nowi += (1 << (len(targs) - 1 - kk))
        datas[i] = nowi
    for i in targs:
        xor = xor ^ (1 << (q_len - 1 - i))
    for i in range(n):
        nowi = datas[i]
        for j in range(n):
            nowj = datas[j]
            if (i & xor) != (j & xor):
                continue
            new_values[i][j] = matrix[nowi][nowj]
    return new_values
