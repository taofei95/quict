#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/24 9:56 下午
# @Author  : Han Yu
# @File    : _ETChecker.py

import random
from QuICT.models import *
from QuICT.algorithm import circuit2circuit, SyntheticalUnitary, Amplitude
from math import pi
import numpy as np

class ETCheckerModel(object):

    @staticmethod
    def getRandomList(l, n):
        _rand = [i for i in range(n)]
        for i in range(n - 1, 0, -1):
            do_get = random.randint(0, i)
            _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
        return _rand[:l]

    @staticmethod
    def cmp_array(arr0, arr1, eps = 1e-6):
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
        self.typeList = _typeList

    def setQubitNumber(self, min_number, max_number):
        """
        随机产生[min_number, max_number]条wire
        :param min_number: qubit 最小随机值
        :param max_number: qubit 最大随机值
        :raise 参数不合法
        """
        if min_number > max_number:
            raise Exception("参数不合法,min_number应小于max_number")
        if min_number == 0 or max_number == 0:
            raise Exception("参数不应为0")
        self.min_qubit_number = min_number
        self.max_qubit_number = max_number

    def setDepthNumber(self, min_number, max_number):
        if min_number > max_number:
            raise Exception("参数不合法,min_number应小于max_number")
        if min_number == 0 or max_number == 0:
            raise Exception("参数不应为0")
        self.min_depth_number = min_number
        self.max_depth_number = max_number

    def setRoundNumber(self, _round_number):
        if _round_number == 0:
            raise Exception("参数不应为0")
        self.round_number = _round_number

    def setAlgorithm(self, _algorithm):
        if not issubclass(_algorithm, circuit2circuit):
            raise Exception("给出的算法不是c2c算法")
        self.algorithm = _algorithm

    def __init__(self):
        self.typeList = []
        self.min_qubit_number = self.max_qubit_number = 0
        self.min_depth_number = self.max_depth_number = 0
        self.round_number = 0
        self.algorithm = None

    def small_circuit_run(self, rand_qubit, rand_depth):
        circuit = Circuit(rand_qubit)

        for _ in range(rand_depth):
            rand_type = random.randrange(0, len(self.typeList))
            GateBuilder.setGateType(self.typeList[rand_type])

            if self.typeList[rand_type] == GateType.Perm:
                # rand = random.randint(1, rand_qubit)
                rand = rand_qubit
                perm_list = [i for i in range(1 << rand)]
                random.shuffle(perm_list)
                tclist = self.getRandomList(rand, rand_qubit)
                GateBuilder.setTargs(tclist)
                GateBuilder.setPargs(perm_list)
                gate = GateBuilder.getGate()
                circuit.gates.append(gate)
            elif self.typeList[rand_type] == GateType.Custom:
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

    def big_circuit_run(self, rand_qubit, rand_depth):
        circuit = Circuit(rand_qubit)

        for _ in range(rand_depth):
            rand_type = random.randrange(0, len(self.typeList))
            GateBuilder.setGateType(self.typeList[rand_type])

            if self.typeList[rand_type] == GateType.Perm:
                rand = random.randint(1, rand_qubit)
                perm_list = [i for i in range(1 << rand)]
                random.shuffle(perm_list)
                tclist = self.getRandomList(rand, rand_qubit)
                GateBuilder.setTargs(tclist)
                GateBuilder.setPargs(perm_list)
                gate = GateBuilder.getGate()
                circuit.gates.append(gate)
            elif self.typeList[rand_type] == GateType.Custom:
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
            circuit.reset_initial_values()
            result_circuit.force_copy(circuit)
            circuit.set_flush_gates(originGates)
            result_circuit.set_flush_gates(resultGates)
            origin = Amplitude.run(circuit)
            result = Amplitude.run(result_circuit)
            if not self.cmp_array(origin, result):
                circuit.print_infomation()
                result_circuit.print_infomation()
                return False
        return True

    def run(self):
        if len(self.typeList) == 0:
            raise Exception("请设置typeList(门列表)")
        if self.min_qubit_number == 0:
            raise Exception("请设置电路比特数")
        if self.min_depth_number == 0:
            raise Exception("请设置电路深度")
        if self.round_number == 0:
            raise Exception("请设置轮数")
        if self.algorithm is None:
            raise Exception("请设置算法")

        for i in range(self.round_number):
            rand_qubit = random.randint(self.min_qubit_number, self.max_qubit_number)
            rand_depth = random.randint(self.min_depth_number, self.max_depth_number)
            if rand_qubit <= 7:
                result = self.small_circuit_run(rand_qubit, rand_depth)
            else:
                result = self.big_circuit_run(rand_qubit, rand_depth)
            if result:
                print("通过第{}轮测试".format(i))
            else:
                print("在第{}轮测试发生错误".format(i))
                return
        print("通过全部测试")

ETChecker = ETCheckerModel()

'''
class ETAncillaeCheckerModel(object):

    @staticmethod
    def getRandomList(l, n):
        _rand = [i for i in range(n)]
        for i in range(n - 1, 0, -1):
            do_get = random.randint(0, i)
            _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
        return _rand[:l]

    @staticmethod
    def cmp_array(arr0, arr1, eps = 1e-6):
        arr0 = np.array(arr0).flatten()
        arr1 = np.array(arr1).flatten()
        if len(arr0) != len(arr1):
            return False
        for i in range(len(arr0)):
            if abs(arr0[i] - arr1[i]) > eps:
                return False
        return True

    def setTypeList(self, _typeList):
        self.typeList = _typeList

    def setQubitNumber(self, min_number, max_number):
        """
        随机产生[min_number, max_number]条wire
        :param min_number: qubit 最小随机值
        :param max_number: qubit 最大随机值
        :raise 参数不合法
        """
        if min_number > max_number:
            raise Exception("参数不合法,min_number应小于max_number")
        if min_number == 0 or max_number == 0:
            raise Exception("参数不应为0")
        self.min_qubit_number = min_number
        self.max_qubit_number = max_number

    def setDepthNumber(self, min_number, max_number):
        if min_number > max_number:
            raise Exception("参数不合法,min_number应小于max_number")
        if min_number == 0 or max_number == 0:
            raise Exception("参数不应为0")
        self.min_depth_number = min_number
        self.max_depth_number = max_number

    def setRoundNumber(self, _round_number):
        if _round_number == 0:
            raise Exception("参数不应为0")
        self.round_number = _round_number

    def setAlgorithm(self, _algorithm):
        if not issubclass(_algorithm, circuit2circuit):
            raise Exception("给出的算法不是c2c算法")
        self.algorithm = _algorithm

    def setMeasureF(self, _measureF):
        self.measureF = _measureF

    def __init__(self):
        self.typeList = []
        self.min_qubit_number = self.max_qubit_number = 0
        self.min_depth_number = self.max_depth_number = 0
        self.round_number = 0
        self.algorithm = None
        self.measureF = None

    def big_circuit_run(self, rand_qubit, rand_depth):
        ancillae = self.measureF(rand_qubit)
        circuit = Circuit(rand_qubit)
        for _ in range(rand_depth):
            rand_type = random.randrange(0, len(self.typeList))
            GateBuilder.setGateType(self.typeList[rand_type])

            if self.typeList[rand_type] == GateType.Perm:
                pass
            elif self.typeList[rand_type] == GateType.Custom:
                pass
            else:
                targs = GateBuilder.getTargsNumber()
                cargs = GateBuilder.getCargsNumber()
                pargs = GateBuilder.getParamsNumber()

                tclist = self.getRandomList(targs + cargs, rand_qubit)
                if targs != 0:
                    GateBuilder.setTargs(tclist[:targs])
                if cargs != 0:
                    GateBuilder.setCargs(tclist[cargs:])
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
            circuit.reset_initial_values()
            result_circuit.force_copy(circuit)
            circuit.set_flush_gates(originGates)
            result_circuit.set_flush_gates(resultGates)
            origin = Amplitude.run(circuit)
            result = Amplitude.run(result_circuit, ancillae)
            if not self.cmp_array(origin, result):
                circuit.print_infomation()
                result_circuit.print_infomation()
                return False
        return True

    def run(self):
        if len(self.typeList) == 0:
            raise Exception("请设置typeList(门列表)")
        if self.min_qubit_number == 0:
            raise Exception("请设置电路比特数")
        if self.min_depth_number == 0:
            raise Exception("请设置电路深度")
        if self.round_number == 0:
            raise Exception("请设置轮数")
        if self.algorithm is None:
            raise Exception("请设置算法")
        if self.measureF is None:
            raise Exception("请设置辅助位")

        for i in range(self.round_number):
            rand_qubit = random.randint(self.min_qubit_number, self.max_qubit_number)
            rand_depth = random.randint(self.min_depth_number, self.max_depth_number)

            result = self.big_circuit_run(rand_qubit, rand_depth)
            if result:
                print("通过第{}轮测试".format(i))
            else:
                print("在第{}轮测试发生错误".format(i))
        print("通过全部测试")

ETAncillaeChecker = ETAncillaeCheckerModel()

'''