#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39 下午
# @Author  : Han Yu
# @File    : _synthesis.py

from QuICT.models import GateBuilder, Qubit, Qureg, Circuit, GateType
from QuICT.exception import TypeException

class Synthesis(object):
    # 门对应控制位数
    @property
    def controls(self) -> int:
        return self.__controls

    @controls.setter
    def controls(self, controls: int):
        self.__controls = controls

    # 门对应控制位索引
    @property
    def cargs(self):
        """
        :return:
            返回一个list，表示控制位
        """
        return self.__cargs

    @cargs.setter
    def cargs(self, cargs: list):
        if isinstance(cargs, list):
            self.__cargs = cargs
        else:
            self.__cargs = [cargs]

    # 门对应作用位数
    @property
    def targets(self) -> int:
        return self.__targets

    @targets.setter
    def targets(self, targets: int):
        self.__targets = targets

    # 门对应作用位索引
    @property
    def targs(self):
        """
        :return:
            返回一个list，代表作用位的list
        """
        return self.__targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, list):
            self.__targs = targs
        else:
            self.__targs = [targs]

    # 辅助数组位个数
    @property
    def params(self) -> int:
        return self.__params

    @params.setter
    def params(self, params: int):
        self.__params = params

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

    @property
    def carg(self):
        return self.cargs[0]

    @property
    def targ(self):
        return self.targs[0]

    def __init__(self):
        self.__cargs = []
        self.__targs = []
        self.__pargs = []
        self.__controls = 0
        self.__targets = 0
        self.__params = 0

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

        gates = self.build_gate()
        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.__add_qureg_gate__(gate, qubits)

    def __call__(self, other):
        """
        使用()添加参数
        :param other: 添加的参数
            1) int/float/complex
            2) list<int/float/complex>
            3) tuple<int/float/complex>
        :raise 类型错误
        :return 修改参数后的self
        """
        if self.permit_element(other):
            self.pargs = [other]
        elif isinstance(other, list):
            self.pargs = []
            for element in other:
                if not self.permit_element(element):
                    raise TypeException("int或float或complex", element)
                self.pargs.append(element)
        elif isinstance(other, tuple):
            self.pargs = []
            for element in other:
                if not self.permit_element(element):
                    raise TypeException("int或float或complex", element)
                self.pargs.append(element)
        else:
            raise TypeException("int/float/complex或list<int/float/complex>或tuple<int/float/complex>", other)
        return self

    def build_gate(self):
        GateBuilder.setGateType(GateType.ID)
        GateBuilder.setTargs(0)
        return [GateBuilder.getGate()]
