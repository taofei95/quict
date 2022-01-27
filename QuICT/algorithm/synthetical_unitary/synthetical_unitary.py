#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:39
# @Author  : Han Yu
# @File    : SyntheticalUnitary.py

import numpy as np

from QuICT.algorithm import Algorithm
from QuICT.core import Circuit


class SyntheticalUnitary(Algorithm):
    @classmethod
    def run(cls, circuit: Circuit, showSU=False):
        """
        get the unitary matrix of the circuit

        Args:
            circuit(Circuit)
            showSU(bool): whether return an SU unitary
        """
        matrix = np.eye(1 << len(circuit.qubits), dtype=np.complex128)
        for gate in circuit.gates:
            if gate.controls + gate.targets == 0:
                continue

            matrix = np.matmul(circuit.matrix_product_to_circuit(gate), matrix)

        if showSU:
            det = np.linalg.det(matrix)
            n = np.shape(matrix)[0]
            det = np.power(det, 1 / n)
            matrix[:] /= det

        return matrix
