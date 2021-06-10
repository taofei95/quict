#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/14 8:38
# @Author  : Han Yu
# @File    : _efficiencyChecker.py

from math import *
import random
import time

import numpy as np

from QuICT.core import *

class StandardEfficiencyCheckerModel(object):
    """ this model is to check the efficiency of amplitude calculating

    Attributes:
        min_qubits(int): the min number of qubits when testing
        max_qubits(int): the max number of qubits when testing
        size(int): the size of circuit for testing
        typeList(list<BasicGate>): the gate which participates in testing

    """

    @staticmethod
    def getRandomList(l, n):
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

    @property
    def min_qubits(self):
        return self.__min_qubits

    @min_qubits.setter
    def min_qubits(self, min_qubits):
        if min_qubits < 10:
            min_qubits = 10
        self.__min_qubits = min_qubits

    @property
    def max_qubits(self):
        return self.__max_qubits

    @max_qubits.setter
    def max_qubits(self, max_qubits):
        if max_qubits > 40:
            max_qubits = 40
        self.__max_qubits = max_qubits

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, size):
        self.__size = size

    @property
    def typeList(self):
        return self.__typeList

    @typeList.setter
    def typeList(self, typeList):
        self.__typeList = typeList

    def __init__(self):
        self.__min_qubits = 10
        self.__max_qubits = 30
        self.__size = 1000
        self.__typeList = [
            GATE_ID['H'],
            GATE_ID['S'],
            GATE_ID['S_dagger'],
            GATE_ID['X'],
            GATE_ID['Y'],
            GATE_ID['Z'],
            GATE_ID['ID'],
            GATE_ID['U1'],
            GATE_ID['U2'],
            GATE_ID['Rx'],
            GATE_ID['Ry'],
            GATE_ID['Rz'],
            GATE_ID['T'],
            GATE_ID['T_dagger'],
            GATE_ID['CZ'],
            GATE_ID['CX'],
            GATE_ID['CH'],
            GATE_ID['CRz'],
            GATE_ID['CCX'],
            GATE_ID['Swap'],
        ]

    def run(self):
        """ run the test

        """
        max_round = 20
        for qubit in range(self.min_qubits, self.max_qubits + 1):
            min_t = -1
            max_t = -1
            ava = 0

            min_tt = -1
            max_tt = -1
            ava_t = 0
            for _ in range(1, max_round):
                gates = []
                for _ in range(0, self.size):
                    rand = random.randrange(0, len(self.typeList))
                    gateType = self.typeList[rand]
                    GateBuilder.setGateType(gateType)
                    targs = GateBuilder.getTargsNumber()
                    cargs = GateBuilder.getCargsNumber()
                    pargs = GateBuilder.getParamsNumber()
                    tclist = self.getRandomList(targs + cargs, qubit)
                    if targs != 0:
                        GateBuilder.setTargs(tclist[:targs])
                    if cargs != 0:
                        GateBuilder.setCargs(tclist[targs:])
                    if pargs != 0:
                        params = []
                        for _ in range(pargs):
                            params.append(random.uniform(0, 2 * pi))
                        GateBuilder.setPargs(params)
                    gate = GateBuilder.getGate()
                    gates.append(gate)

                circuit = Circuit(qubit)
                circuit.set_exec_gates(gates)
                time_start = time.time()
                circuit.exec()
                time_end = time.time()
                time_amplitude = np.round(time_end - time_start, decimals=3)
                del circuit
                ava = ava + time_amplitude / max_round
                if min_t == -1:
                    min_t = time_amplitude
                else:
                    min_t = min(min_t, time_amplitude)

                max_t = max(max_t, time_amplitude)

                circuit = Circuit(qubit)
                # circuit.reset_initial_zeros()
                circuit.assign_initial_zeros()
                circuit.set_exec_gates(gates)
                time_start = time.time()
                circuit.exec()
                time_end = time.time()
                time_amplitude = np.round(time_end - time_start, decimals=3)
                del circuit
                ava_t = ava_t + time_amplitude / max_round
                if min_tt == -1:
                    min_tt = time_amplitude
                else:
                    min_tt = min(min_tt, time_amplitude)

                max_tt = max(max_tt, time_amplitude)

            print(f'{np.round(ava_t, decimals=2)}({np.round(ava, decimals=2)})'
                  f'{np.round(min_tt, decimals=2)}({np.round(min_t, decimals=2)}) '
                  f'{np.round(max_tt, decimals=2)}({np.round(max_t, decimals=2)})')

StandardEfficiencyChecker = StandardEfficiencyCheckerModel()


