#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:37
# @Author  : Han Yu
# @File    : Amplitude.py

from .._algorithm import Algorithm
from QuICT import *
from QuICT.backends import systemCdll
import numpy as np
from ctypes import c_int

class Amplitude(Algorithm):
    """ get the amplitude of some circuit with some ancillary qubits which are ignored

    """

    @classmethod
    def run(cls, circuit : Circuit, ancilla = None):
        """

        Args:
            circuit(Circuit)
            ancilla(list<int>): the indexes of ancillary qubits
        """
        if ancilla is None:
            ancilla = []
        for qubit_index in ancilla:
            Measure | circuit(qubit_index)
        circuit.exec()
        for qubit_index in ancilla:
            if int(circuit(qubit_index)) == 1:
                raise Exception("the ancillary is not 0")
        return cls._run(circuit, ancilla)

    @staticmethod
    def _run(circuit: Circuit, ancilla = None):
        """

        Args:
            circuit(Circuit)
            ancilla(list<int>): the indexes of ancillary qubits
        """
        circuit.exec()
        dll = systemCdll.quick_operator_cdll
        amplitude_cheat_operator = dll.amplitude_cheat_operator
        amplitude_cheat_operator.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
            c_int,
            np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS"),
        ]

        length = 1 << (circuit.circuit_width() - len(ancilla))

        amplitude_cheat_operator.restype = np.ctypeslib.ndpointer(dtype=np.complex128, shape=(length,))

        tangle_list = []
        tangle_values = np.array([], dtype=np.complex128)
        tangle_length = np.array([], dtype=np.int64)
        qubit_map     = np.array([i for i in range(circuit.circuit_width() - len(ancilla))], dtype=np.int64)

        tangle_iter = 0
        q_index = 0
        for qubit in circuit.qubits:
            if q_index in ancilla:
                q_index += 1
                continue
            q_index += 1
            if qubit.qState not in tangle_list:
                tangle_list.append(qubit.qState)
        for tangle in tangle_list:
            tangle_values = np.append(tangle_values, tangle.values)
            tangle_length = np.append(tangle_length, len(tangle.qureg))
            for q in tangle.qureg:
                qubit_map[tangle_iter + tangle.index_for_qubit(q)] = \
                    q.circuit.index_for_qubit(q, ancilla)
            tangle_iter = tangle_iter + len(tangle.qureg)
        tangle_length = np.array(tangle_length, dtype=np.int64)
        tangle_values = np.array(tangle_values, dtype=np.complex128)
        ndpointer = amplitude_cheat_operator(
            tangle_values,
            tangle_length,
            len(tangle_list),
            qubit_map
        )
        values = np.ctypeslib.as_array(ndpointer, shape=(length,))
        return values.tolist()
