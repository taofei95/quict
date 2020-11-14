#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _synthesis.py

from QuICT.models import GateBuilder, Qubit, Qureg, Circuit, GateType
from QuICT.exception import TypeException

class Synthesis(object):
    # 辅助数组数组
    @property
    def pargs(self):
        """
        :return:
            返回一个list，代表辅助数组
        """
        return self.__pargs

    @pargs.setter
    def pargs(self, pargs: list):
        if isinstance(pargs, list):
            self.__pargs = pargs
        else:
            self.__pargs = [pargs]

    @property
    def parg(self):
        return self.pargs[0]

    def __init__(self):
        self.__pargs = []

    # 作用位数量
    @property
    def targets(self):
        """
        :return:
            返回一个作用位数量
        """
        return self.__targets

    @targets.setter
    def targets(self, targets):
        self.__targets = targets

    def __init__(self):
        self.__pargs = []
        self.__targets = 0

    @staticmethod
    def qureg_trans(other):
        """
        将输入转化为标准Qureg
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :return Qureg
        :raise TypeException
        """
        if isinstance(other, tuple):
            other = list(other)
        if isinstance(other, list):
            qureg = Qureg()
            for item in other:
                if isinstance(item, Qubit):
                    qureg.append(item)
                elif isinstance(item, Qureg):
                    qureg.extend(item)
                else:
                    raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit或tuple<qubit>或qureg或circuit", other)
        return qureg

    @staticmethod
    def permit_element(element):
        """
        参数只能为int/float/complex
        :param element: 待判断的元素
        :return: 是否允许作为参数
        :raise 不允许的参数
        """
        if isinstance(element, int) or isinstance(element, float) or isinstance(element, complex):
            return True
        else:
            return False

    def __or__(self, other):
        """
        处理作用于多个位或带有参数的门
        :param other: 作用的对象
            1）tuple<qubit, qureg>
            2) qureg/list<qubit, qureg>
            3) Circuit
        """
        if isinstance(other, tuple):
            other = list(other)
        if isinstance(other, list):
            qureg = Qureg()
            for item in other:
                if isinstance(item, Qubit):
                    qureg.append(item)
                elif isinstance(item, Qureg):
                    qureg.extend(item)
                else:
                    raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit或tuple<qubit>或qureg或circuit", other)

        self.targets = len(qureg)

        gates = self.build_gate()
        if isinstance(gates, Circuit):
            gates = gates.gates
        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.__add_qureg_gate__(gate, qubits)

    def __xor__(self, other):
        """
        处理作用于多个位或带有参数的门
        :param other: 作用的对象
            1）tuple<qubit, qureg>
            2) qureg/list<qubit, qureg>
            3) Circuit
        """
        if isinstance(other, tuple):
            other = list(other)
        if isinstance(other, list):
            qureg = Qureg()
            for item in other:
                if isinstance(item, Qubit):
                    qureg.append(item)
                elif isinstance(item, Qureg):
                    qureg.extend(item)
                else:
                    raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit或tuple<qubit>或qureg或circuit", other)

        gates = self.build_gate()
        if isinstance(gates, Circuit):
            gates = gates.gates
        gates = GateBuilder.reflect_gates(gates)
        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.__add_qureg_gate__(gate, qubits)

    def __call__(self, *pargs):
        raise Exception("请重写__call__方法")

    def build_gate(self):
        raise Exception("请重写build_gate方法")
