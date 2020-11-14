#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/11 10:33 下午
# @Author  : Han Yu
# @File    : _circuit2param.py

from QuICT.models import Circuit, Measure
from QuICT.backends import systemCdll
import numpy as np
from ctypes import *

class circuit2param(object):
    @classmethod
    def run(cls, circuit: Circuit):
        """
        :param circuit:  待处理电路
        :return: 返回参数
        """
        circuit.const_lock = True
        params = cls.__run__(circuit)
        circuit.const_lock = False
        return params

    @staticmethod
    def __run__(circuit: Circuit):
        """
        需要其余算法改写
        :param circuit: 待处理电路
        :return: 返回参数
        """
        return circuit.const_lock

class Amplitude(circuit2param):

    @classmethod
    def run(cls, circuit : Circuit, ancilla = None):
        if ancilla is None:
            ancilla = []
        for qubit_index in ancilla:
            Measure | circuit(qubit_index)
        for qubit_index in ancilla:
            if circuit(qubit_index) == 1:
                raise Exception("辅助位不为0")
        return cls.__run__(circuit, ancilla)

    @staticmethod
    def __run__(circuit: Circuit, ancilla = None):
        """
        需要其余算法改写
        :param circuit: 待处理电路
        :param ancilla: 无需计算的list
        :return: 返回list
        """
        circuit.flush()
        dll = systemCdll.quick_operator_cdll
        amplitude_cheat_operator = dll.amplitude_cheat_operator
        amplitude_cheat_operator.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
            c_int,
            np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
        ]

        length = 1 << (circuit.circuit_length() - len(ancilla))

        amplitude_cheat_operator.restype = np.ctypeslib.ndpointer(dtype=np.complex, shape=(length,))

        tangle_list = []
        tangle_values = np.array([], dtype=np.complex)
        tangle_length = np.array([], dtype=np.int)
        qubit_map     = np.array([i for i in range(circuit.circuit_length() - len(ancilla))], dtype=np.int)

        tangle_iter = 0
        q_index = 0
        for qubit in circuit.qubits:
            if q_index in ancilla:
                q_index += 1
                continue
            q_index += 1
            if qubit.tangle not in tangle_list:
                tangle_list.append(qubit.tangle)
                '''
                tangle_values = np.append(tangle_values, qubit.tangle.values)
                tangle_length = np.append(tangle_length, len(qubit.tangle.qureg))
                for q in qubit.tangle.qureg:
                    qubit_map[tangle_iter + qubit.tangle.index_for_qubit(q)] = qubit.circuit.index_for_qubit(q)
                tangle_iter = tangle_iter + len(qubit.tangle.qureg)
                '''
        for tangle in tangle_list:
            tangle_values = np.append(tangle_values, tangle.values)
            tangle_length = np.append(tangle_length, len(tangle.qureg))
            for q in tangle.qureg:
                qubit_map[tangle_iter + tangle.index_for_qubit(q)] = \
                    q.circuit.index_for_qubit(q, ancilla)
            tangle_iter = tangle_iter + len(tangle.qureg)
        ndpointer = amplitude_cheat_operator(
            tangle_values,
            tangle_length,
            len(tangle_list),
            qubit_map
        )
        values = np.ctypeslib.as_array(ndpointer, shape=(length,))
        values = np.round(values, decimals=6)
        return values.tolist()

class SyntheticalUnitary(circuit2param):
    @classmethod
    def run(cls, circuit: Circuit, showSU = True):
        """
        :param showSU: 展示SU
        :param circuit:  待处理电路
        :return: 返回参数
        """
        circuit.const_lock = True
        params = cls.__run__(circuit)
        circuit.const_lock = False
        return params

    @staticmethod
    def __run__(circuit: Circuit, showSU = True):
        """
        需要其余算法改写
        :param circuit: 待处理电路
        :return: 返回list
        """
        matrix = np.eye(1 << len(circuit.qubits))
        for gate in circuit.gates:
            if gate.is_measure():
                continue
            matrix = np.matmul(circuit.matrix_product_to_circuit(gate), matrix)
        if showSU:
            det = np.linalg.det(matrix)
            n = np.shape(matrix)[0]
            det = np.power(det, 1 / n)
            matrix[:] /= det
        matrix = np.round(matrix, decimals=6)
        return np.asmatrix(matrix)
