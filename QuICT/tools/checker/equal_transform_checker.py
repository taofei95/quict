#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/24 9:56
# @Author  : Han Yu
# @File    : _ETChecker.py

from math import pi
import random

import numpy as np

from QuICT.algorithm import SyntheticalUnitary, Amplitude
from QuICT.core import *

class ETCheckerModel(object):
    """ checker whether the input and output(circuit) of a algorithm is equivalence

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

    @staticmethod
    def cmp_array(arr0, arr1, eps = 1e-6):
        """ compare two array with eps
        Args:
            arr0(list<complex>): compare array 0
            arr1(list<complex>): compare array 1
            eps(float): permitted epsilon
        Returns:
            bool: True if arr0 is equal to arr1 with permitted epsilon
        """
        arr0 = np.array(arr0).flatten()
        arr1 = np.array(arr1).flatten()
        if len(arr0) != len(arr1):
            return False
        for i in range(len(arr0)):
            if abs(arr0[i] - arr1[i]) > eps:
                print(i, arr0[i], arr1[i], i // 64, i % 64)
                return False
        return True

    def setTypeList(self, _typeList):
        """ Set the checker type list

        Args:
            _typeList(list<BasicGate>): types for testing

        """
        self.typeList = _typeList

    def setQubitNumber(self, min_number, max_number):
        """ set the min and max number of qubits of circuit

        Args:
            min_number(int): min number of qubits
            max_number(int): max number of qubits
        """
        if min_number > max_number:
            raise Exception("parameter is error, min_number shouldn't greater than max_number")
        if min_number == 0 or max_number == 0:
            raise Exception("parameters shouln't be 0")
        self.min_qubit_number = min_number
        self.max_qubit_number = max_number

    def setSize(self, min_number, max_number):
        """ set the min and max number of size of circuit

        Args:
            min_number(int): min number of size
            max_number(int): max number of size
        """
        if min_number > max_number:
            raise Exception("parameter is error, min_number shouldn't greater than max_number")
        if min_number == 0 or max_number == 0:
            raise Exception("parameters shouln't be 0")
        self.min_size_number = min_number
        self.max_size_number = max_number

    def setRoundNumber(self, _round_number):
        """ set the min and max number of test round

        Args:
            _round_number(int): round number
        """
        if _round_number == 0:
            raise Exception("parameter shouln't be 0")
        self.round_number = _round_number

    def setAlgorithm(self, _algorithm):
        """ set the algorithm to be tested

        Args:
            _algorithm: the algorithm to be tested
        """
        self.algorithm = _algorithm

    def __init__(self):
        self.typeList = []
        self.min_qubit_number = self.max_qubit_number = 0
        self.min_size_number = self.max_size_number = 0
        self.round_number = 0
        self.algorithm = None

    def small_circuit_run(self, rand_qubit, rand_size):
        """ if the test circuit is small, get the matrix of circuit

        Args:
            rand_qubit(int): number of qubits of the circuit
            rand_size(int): number of size of the circuit
        Returns:
            bool: True if pass the test
        """
        circuit = Circuit(rand_qubit)

        for _ in range(rand_size):
            rand_type = random.randrange(0, len(self.typeList))
            GateBuilder.setGateType(self.typeList[rand_type])

            if self.typeList[rand_type] == GATE_ID["Perm"]:
                # rand = random.randint(1, rand_qubit)
                rand = rand_qubit
                perm_list = [i for i in range(1 << rand)]
                random.shuffle(perm_list)
                tclist = self.getRandomList(rand, rand_qubit)
                GateBuilder.setTargs(tclist)
                GateBuilder.setPargs(perm_list)
                gate = GateBuilder.getGate()
                circuit.gates.append(gate)
            elif self.typeList[rand_type] == GATE_ID["Unitary"]:
                pass
            else:
                targs = GateBuilder.getTargsNumber()
                cargs = GateBuilder.getCargsNumber()
                pargs = GateBuilder.getParamsNumber()

                tclist = self.getRandomList(targs + cargs, rand_qubit)
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
                circuit.gates.append(gate)

        origin = SyntheticalUnitary.run(circuit)
        result_circuit = self.algorithm.run(circuit)
        result = SyntheticalUnitary.run(result_circuit)
        if self.cmp_array(origin, result):
            return True
        else:
            circuit.print_infomation()
            result_circuit.print_infomation()
            return False

    def big_circuit_run(self, rand_qubit, rand_size):
        """ if the test circuit is small, get the amplitude of circuit

        generate random initial values for circuits and test for some times

        Args:
            rand_qubit(int): number of qubits of the circuit
            rand_size(int): number of size of the circuit
        Returns:
            bool: True if pass the test
        """
        circuit = Circuit(rand_qubit)

        for _ in range(rand_size):
            rand_type = random.randrange(0, len(self.typeList))
            GateBuilder.setGateType(self.typeList[rand_type])

            if self.typeList[rand_type] == GATE_ID["Perm"]:
                rand = random.randint(1, rand_qubit)
                perm_list = [i for i in range(1 << rand)]
                random.shuffle(perm_list)
                tclist = self.getRandomList(rand, rand_qubit)
                GateBuilder.setTargs(tclist)
                GateBuilder.setPargs(perm_list)
                gate = GateBuilder.getGate()
                circuit.gates.append(gate)
            elif self.typeList[rand_type] == GATE_ID["Unitary"]:
                pass
            else:
                targs = GateBuilder.getTargsNumber()
                cargs = GateBuilder.getCargsNumber()
                pargs = GateBuilder.getParamsNumber()
                tclist = self.getRandomList(targs + cargs, rand_qubit)
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
                circuit.gates.append(gate)
        result_circuit = self.algorithm.run(circuit)
        originGates = circuit.gates
        resultGates = result_circuit.gates
        for _ in range(64):
            circuit.assign_initial_random()
            result_circuit.force_copy(circuit)
            circuit.set_exec_gates(originGates)
            result_circuit.set_exec_gates(resultGates)
            origin = Amplitude.run(circuit)
            result = Amplitude.run(result_circuit)
            if not self.cmp_array(origin, result):
                circuit.print_infomation()
                result_circuit.print_infomation()
                return False
        return True

    def run(self):
        if len(self.typeList) == 0:
            raise Exception("please set the typelist")
        if self.min_qubit_number == 0:
            raise Exception("please set the number of qubits")
        if self.min_size_number == 0:
            raise Exception("please set the size of circuit")
        if self.round_number == 0:
            raise Exception("please set the test round")
        if self.algorithm is None:
            raise Exception("please set the test algorithm")

        for i in range(self.round_number):
            rand_qubit = random.randint(self.min_qubit_number, self.max_qubit_number)
            rand_size = random.randint(self.min_size_number, self.max_size_number)
            if rand_qubit <= 7:
                result = self.small_circuit_run(rand_qubit, rand_size)
            else:
                result = self.big_circuit_run(rand_qubit, rand_size)
            if result:
                print(f"pass the {i} round test")
            else:
                print(f"failed in the {i} round test")
                return
        print("pass all test")

ETChecker = ETCheckerModel()
