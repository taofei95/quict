#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:39
# @Author  : Han Yu
# @File    : SyntheticalUnitary.py

import numpy as np

from QuICT.algorithm import Algorithm
from QuICT.core import Circuit


class SyntheticalUnitary(Algorithm):
    """ get the unitary matrix of the circuit


    """
    @classmethod
    def run(cls, circuit: Circuit, showSU=False):
        """

        Args:
            circuit(Circuit)
            showSU(bool): whether return an SU unitary
        """
        circuit.const_lock = True
        params = cls.__run__(circuit, showSU)
        circuit.const_lock = False
        return params

    @staticmethod
    def __run__(circuit: Circuit, showSU=True):
        """

        Args:
            circuit(Circuit)
            showSU(bool): whether return an SU unitary
        """
        matrix = np.eye(1 << len(circuit.qubits))
        for gate in circuit.gates:
            if gate.qasm_name == "measure":
                continue
            matrix = np.matmul(circuit.matrix_product_to_circuit(gate), matrix)
        if showSU:
            det = np.linalg.det(matrix)
            n = np.shape(matrix)[0]
            det = np.power(det, 1 / n)
            matrix[:] /= det
        matrix = matrix
        return np.array(matrix)
