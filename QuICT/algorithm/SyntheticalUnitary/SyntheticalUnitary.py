#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:39 上午
# @Author  : Han Yu
# @File    : SyntheticalUnitary.py

from .._algorithm import Algorithm
from QuICT.models import *
import numpy as np

class SyntheticalUnitary(Algorithm):
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
