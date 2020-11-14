#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:04 下午
# @Author  : Han Yu
# @File    : _gate.py

from ._qubit import Qubit, Qureg
from ._circuit import Circuit
from QuICT.exception import TypeException
from enum import Enum
# from math import sqrt, cos, sin, pi, log2, gcd, ceil
import numpy as np
import copy

class GateType(Enum):
    Error = -1
    H = 0
    S = 1
    S_dagger = 2
    X = 3
    Y = 4
    Z = 5
    ID = 6
    U0 = 7
    U1 = 8
    U2 = 9
    U3 = 10
    Rx = 11
    Ry = 12
    Rz = 13
    T = 14
    T_dagger = 15

    CZ = 16
    CX = 17
    CY = 18
    CH = 19
    CRz = 20

    CCX = 21

    Measure = 22
    Reset = 23
    Barrier = 24

    Swap = 25

    Perm = 26
    Custom = 27
    ControlPermMulDetail = 28
    ShorInital = 29

class BasicGate(object):
    """
    类的属性
    """

    # 门对应的矩阵
    @property
    def matrix(self) -> list:
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix: list):
        self.__matrix = matrix

    def compute_matrix(self):
        return self.matrix

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

    @property
    def qasm_name(self):
        return self.__qasm_name

    @qasm_name.setter
    def qasm_name(self, qasm_name):
        self.__qasm_name = qasm_name

    def __init__(self):
        self.__matrix = []
        self.__cargs = []
        self.__targs = []
        self.__pargs = []
        self.__controls = 0
        self.__targets = 0
        self.__params = 0
        self.__qasm_name = 'error'

    def __or__(self, other):
        """
        处理｜运算符
        单qubit门作用到电路或者qureg上，视为对所有qubit各自作用该门
        :param other: 作用的对象
            1）tuple<qubit, qureg>
            2) qureg/list<qubit, qureg>
            3) Circuit
        :raise other类型错误
        """
        if self.is_single():
            self.or_deal_single(other)
        else:
            self.or_deal_other(other)

    @staticmethod
    def type():
        """
        :return: 返回H
        """
        return GateType.Error

    def is_single(self) -> bool:
        return 0 <= self.type().value  <= 15

    def is_control_single(self) -> bool:
        return (self.type().value >= 16) and (self.type().value <= 20)

    def is_diagonal(self) -> bool:
        if not self.is_single() and not self.is_control_single() and not self.is_swap():
            raise Exception("只考虑单比特门和基础双比特门")
        if self.is_single():
            matrix = self.matrix
            if abs(matrix[1]) < 1e-10 and abs(matrix[2]) < 1e-10:
                return True
            else:
                return False
        elif self.is_control_single():
            matrix = self.matrix
            if abs(matrix[1]) < 1e-10 and abs(matrix[2]) < 1e-10:
                return True
            else:
                return False
        else:
            return False

    def is_ccx(self) -> bool:
        return self.type() == GateType.CCX

    def is_measure(self) -> bool:
        return self.type() == GateType.Measure

    def is_reset(self) -> bool:
        return self.type() == GateType.Reset

    def is_swap(self) -> bool:
        return self.type() == GateType.Swap

    def is_perm(self) -> bool:
        return self.type() == GateType.Perm

    def is_custom(self) -> bool:
        return self.type() == GateType.Custom

    def is_shorInit(self) -> bool:
        return self.type() == GateType.ShorInital

    def is_controlMulPer(self) -> bool:
        return self.type() == GateType.ControlPermMulDetail

    def is_barrier(self) -> bool:
        return self.type() == GateType.Barrier

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

    def deal_qubit(self, qubit):
        """
        对单个qubit添加门
        :param qubit: Qubit

        """
        name = str(self.__class__.__name__)
        gate = globals()[name]()
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targets = self.targets
        gate.controls = self.controls
        gate.params = self.params
        qubit.circuit.__add_qubit_gate__(gate, qubit)

    def deal_qureg(self, qureg):
        """
        对qureg添加门
        :param qureg: Qureg
        """
        if self.targets + self.controls != len(qureg):
            raise GateDigitException(self.controls, self.targets, len(qureg))
        name = str(self.__class__.__name__)
        gate = globals()[name]()
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targets = self.targets
        gate.controls = self.controls
        gate.params = self.params
        if isinstance(gate, CustomGate):
            gate.matrix = self.matrix
        qureg.circuit.__add_qureg_gate__(gate, qureg)

    def or_deal_single(self, other):
        """
        处理一个singleGate
        :param other: 作用的对象
            1）qubit/tuple<qubit>
            2) qureg
            3) Circuit
        :raise other类型错误
        """
        if isinstance(other, tuple):
            for qubit in other:
                if not isinstance(qubit, Qubit):
                    raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", other)
                self.deal_qubit(qubit)
        elif isinstance(other, Qubit):
            self.deal_qubit(other)
        elif isinstance(other, Qureg):
            for qubit in other:
                self.deal_qubit(qubit)
        elif isinstance(other, Circuit):
            for qubit in other.qubits:
                self.deal_qubit(qubit)
        else:
            raise TypeException("qubit或tuple<qubit>或qureg或circuit", other)

    def or_deal_other(self, other):
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
        self.deal_qureg(qureg)

    def print_info(self):
        infomation = self.__str__()
        if self.controls != 0:
            infomation = infomation + " 控制位:{} ".format(self.cargs)
        if self.targets != 0:
            infomation = infomation + " 作用位:{} ".format(self.targs)
        if self.params != 0:
            infomation = infomation + " 参数:{} ".format(self.pargs)
        print(infomation)

    def qasm(self):
        if self.qasm_name == 'error':
            return 'error'
        string = self.qasm_name
        if self.params > 0:
            string += '('
            for i in range(len(self.pargs)):
                if i != 0:
                    string += ', '
                string += str(self.pargs[i])
            string += ')'
        string += ' '
        first_in = True
        for p in self.cargs:
            if not first_in:
                string += ', '
            else:
                first_in = False
            string += 'q[{}]'.format(p)
        for p in self.targs:
            if not first_in:
                string += ', '
            else:
                first_in = False
            string += 'q[{}]'.format(p)
        string += ';\n'
        return string

    def inverse(self):
        raise Exception("未定义的逆")

class HGate(BasicGate):
    """
    H门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "h"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1 / np.sqrt(2), 1 / np.sqrt(2),
            1 / np.sqrt(2), -1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "H门"

    @staticmethod
    def type():
        """
        :return: 返回H
        """
        return GateType.H

    def inverse(self):
        _H = HGate()
        _H.targs = copy.deepcopy(self.targs)
        return _H

H = HGate()

class SGate(BasicGate):
    """
    S门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "s"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, 1j
        ], dtype = np.complex)

    def __str__(self):
        return "S门"

    @staticmethod
    def type():
        """
        :return: 返回S
        """
        return GateType.S

    def inverse(self):
        _S_dagger = SDaggerGate()
        _S_dagger.targs = copy.deepcopy(self.targs)
        return _S_dagger

S = SGate()


class SDaggerGate(BasicGate):
    """
    S门的共轭转置门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "sdg"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, -1j
        ], dtype = np.complex)

    def __str__(self):
        return "S门的共轭转置门"

    @staticmethod
    def type():
        """
        :return: 返回S的共轭转置门
        """
        return GateType.S_dagger

    def inverse(self):
        _SBACK = SGate()
        _SBACK.targs = copy.deepcopy(self.targs)
        return _SBACK

S_dagger = SDaggerGate()


class XGate(BasicGate):
    """
    X门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "x"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0, 1,
            1, 0
        ], dtype = np.complex)

    def __str__(self):
        return "X门"

    @staticmethod
    def type():
        """
        :return: 返回X
        """
        return GateType.X

    def inverse(self):
        _X = XGate()
        _X.targs = copy.deepcopy(self.targs)
        return _X

X = XGate()


class YGate(BasicGate):
    """
    Y门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "y"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0, -1j,
            1j, 0
        ], dtype = np.complex)

    def __str__(self):
        return "Y门"

    @staticmethod
    def type():
        """
        :return: 返回Y
        """
        return GateType.Y

    def inverse(self):
        _Y = YGate()
        _Y.targs = copy.deepcopy(self.targs)
        return _Y

Y = YGate()


class ZGate(BasicGate):
    """
    Z门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "z"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, -1
        ], dtype = np.complex)

    def __str__(self):
        return "Z门"

    @staticmethod
    def type():
        """
        :return: 返回Z
        """
        return GateType.Z

    def inverse(self):
        _Z = ZGate()
        _Z.targs = copy.deepcopy(self.targs)
        return _Z

Z = ZGate()


class IDGate(BasicGate):
    """
    ID门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "id"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, 1
        ], dtype = np.complex)

    def __str__(self):
        return "ID门"

    @staticmethod
    def type():
        """
        :return: 返回ID
        """
        return GateType.ID

    def inverse(self):
        _ID = IDGate()
        _ID.targs = copy.deepcopy(self.targs)
        return _ID


ID = IDGate()

class U1Gate(BasicGate):
    """
    U1门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "u1"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, np.exp(1j * self.pargs[0])
        ], dtype = np.complex)

    def __str__(self):
        return "U1门"

    @staticmethod
    def type():
        """
        :return: 返回U1
        """
        return GateType.U1

    def inverse(self):
        _U1 = U1Gate()
        _U1.targs = copy.deepcopy(self.targs)
        _U1.pargs = [-self.pargs[0]]
        return _U1

U1 = U1Gate()

class U2Gate(BasicGate):
    """
    U2门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 2
        self.pargs = [np.pi / 2, np.pi / 2]
        self.qasm_name = "u2"

    @property
    def matrix(self) -> np.ndarray:
        sqrt2 = 1 / np.sqrt(2)
        return np.array([
            1 * sqrt2,
            -np.exp(1j * self.pargs[1]) * sqrt2,
            np.exp(1j * self.pargs[0]) * sqrt2,
            np.exp(1j * (self.pargs[0] + self.pargs[1])) * sqrt2
        ], dtype = np.complex)

    def __str__(self):
        return "U2门"

    @staticmethod
    def type():
        """
        :return: 返回U2
        """
        return GateType.U2

    def inverse(self):
        _U2 = U2Gate()
        _U2.targs = copy.deepcopy(self.targs)
        _U2.pargs = [np.pi - self.pargs[1], np.pi - self.pargs[0]]
        return _U2

U2 = U2Gate()

class U3Gate(BasicGate):
    """
    U3门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 3
        self.pargs = [0, 0, np.pi / 2]
        self.qasm_name = "u3"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            np.cos(self.pargs[0] / 2),
            -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
            np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
            np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)
        ], dtype = np.complex)

    def __str__(self):
        return "U3门"

    @staticmethod
    def type():
        """
        :return: 返回U3
        """
        return GateType.U3

    def inverse(self):
        _U3 = U3Gate()
        _U3.targs = copy.deepcopy(self.targs)
        _U3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]
        return _U3

U3 = U3Gate()

class RxGate(BasicGate):
    """
    Rx门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "rx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            np.cos(self.parg / 2),
            1j * -np.sin(self.parg / 2),
            1j * -np.sin(self.parg / 2),
            np.cos(self.parg / 2),
        ], dtype = np.complex)

    def __str__(self):
        return "Rx门"

    @staticmethod
    def type():
        """
        :return: 返回Rx
        """
        return GateType.Rx

    def inverse(self):
        _Rx = RxGate()
        _Rx.targs = copy.deepcopy(self.targs)
        _Rx.pargs = [-self.pargs[0]]
        return _Rx

Rx = RxGate()

class RyGate(BasicGate):
    """
    Ry门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "ry"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            np.cos(self.pargs[0] / 2),
            -np.sin(self.pargs[0] / 2),
            np.sin(self.pargs[0] / 2),
            np.cos(self.pargs[0] / 2),
        ], dtype = np.complex)

    def __str__(self):
        return "Ry门"

    @staticmethod
    def type():
        """
        :return: 返回Ry
        """
        return GateType.Ry

    def inverse(self):
        _Ry = RyGate()
        _Ry.targs = copy.deepcopy(self.targs)
        _Ry.pargs = [-self.pargs[0]]
        return _Ry

Ry = RyGate()

class RzGate(BasicGate):
    """
    Rz门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "rz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,
            0,
            0,
            np.cos(self.pargs) + 1j * np.sin(self.pargs)
        ], dtype = np.complex)

    def __str__(self):
        return "Rz门"

    @staticmethod
    def type():
        """
        :return: 返回Rz
        """
        return GateType.Rz

    def inverse(self):
        _Rz = RzGate()
        _Rz.targs = copy.deepcopy(self.targs)
        _Rz.pargs = [-self.pargs[0]]
        return _Rz

Rz = RzGate()

class TGate(BasicGate):
    """
    T门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "t"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,
            0,  1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "T门"

    @staticmethod
    def type():
        """
        :return: 返回T
        """
        return GateType.T

    def inverse(self):
        _Tdagger = TDaggerGate()
        _Tdagger.targs = copy.deepcopy(self.targs)
        return _Tdagger

T = TGate()

class TDaggerGate(BasicGate):
    """
    T门的共轭转置
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "tdg"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,
            0,  1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "T门的共轭转置"

    @staticmethod
    def type():
        """
        :return: 返回T_dagger
        """
        return GateType.T_dagger

    def inverse(self):
        _Tgate = TGate()
        _Tgate.targs = copy.deepcopy(self.targs)
        return _Tgate

T_dagger = TDaggerGate()

class CZGate(BasicGate):
    """
    CZ门
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,
            0,  -1
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, -1
        ], dtype = np.complex)

    def __str__(self):
        return "CZ门"

    @staticmethod
    def type():
        """
        :return: 返回CZ
        """
        return GateType.CZ

    def inverse(self):
        _CZ = CZGate()
        _CZ.cargs = copy.deepcopy(self.cargs)
        _CZ.targs = copy.deepcopy(self.targs)
        return _CZ

CZ = CZGate()

class CXGate(BasicGate):
    """
    CX门
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0,  1,
            1,  0
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0
        ], dtype = np.complex)

    def __str__(self):
        return "CX门"

    @staticmethod
    def type():
        """
        :return: 返回CX
        """
        return GateType.CX

    def inverse(self):
        _CX = CXGate()
        _CX.cargs = copy.deepcopy(self.cargs)
        _CX.targs = copy.deepcopy(self.targs)
        return _CX

CX = CXGate()

class CYGate(BasicGate):
    """
    CY门
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cy"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0,  1,
            1,  0
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0
        ], dtype = np.complex)

    def __str__(self):
        return "CY门"

    @staticmethod
    def type():
        """
        :return: 返回CY
        """
        return GateType.CY

    def inverse(self):
        _CY = CYGate()
        _CY.cargs = copy.deepcopy(self.cargs)
        _CY.targs = copy.deepcopy(self.targs)
        return _CY

CY = CYGate()

class CHGate(BasicGate):
    """
    CH门
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "ch"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1 / np.sqrt(2), 1 / np.sqrt(2),
            1 / np.sqrt(2), -1 / np.sqrt(2)
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2),
            0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "CH门"

    @staticmethod
    def type():
        """
        :return: 返回CH
        """
        return GateType.CH

    def inverse(self):
        _CH = CHGate()
        _CH.cargs = copy.deepcopy(self.cargs)
        _CH.targs = copy.deepcopy(self.targs)
        return _CH

CH = CHGate()

class CRzGate(BasicGate):
    """
    CRz门
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 1
        self.qasm_name = "crz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,
            0,
            0,
            np.cos(self.parg) + 1j * np.sin(self.parg)
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, np.cos(self.pargs[0]) + 1j * np.sin(self.pargs[0])
        ], dtype = np.complex)

    def __str__(self):
        return "CRz门"

    @staticmethod
    def type():
        """
        :return: 返回CRz
        """
        return GateType.CRz

    def inverse(self):
        _CRz = CRzGate()
        _CRz.cargs = copy.deepcopy(self.cargs)
        _CRz.targs = copy.deepcopy(self.targs)
        _CRz.pargs = [-self.pargs[0]]
        return _CRz

CRz = CRzGate()

class CCXGate(BasicGate):
    """
    CCX门
    """

    def __init__(self):
        super().__init__()
        self.controls = 2
        self.targets = 1
        self.params = 0
        self.qasm_name = "ccx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0,  1,
            1,  0
        ], dtype = np.complex)

    def compute_matrix(self) -> np.ndarray:
        return np.array([
            1,  0,  0,  0,  0,  0,  0,  0,
            0,  1,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  0,  0,  0,  0,  0,
            0,  0,  0,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  1,  0,  0,  0,
            0,  0,  0,  0,  0,  1,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  1,  0
        ], dtype = np.complex)

    def __str__(self):
        return "CCX门"

    @staticmethod
    def type():
        """
        :return: 返回CCX
        """
        return GateType.CCX

    def inverse(self):
        _CCX = CCXGate()
        _CCX.cargs = copy.deepcopy(self.cargs)
        _CCX.targs = copy.deepcopy(self.targs)
        return _CCX

CCX = CCXGate()

class MeasureGate(BasicGate):
    """
    Measure门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "measure"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("试图获取Measure门对应矩阵")

    def __str__(self):
        return "Measure门"

    @staticmethod
    def type():
        """
        :return: 返回Measure
        """
        return GateType.Measure
Measure = MeasureGate()

class ResetGate(BasicGate):
    """
    Reset门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "reset"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("试图获取Reset门对应矩阵")

    def __str__(self):
        return "Reset门"

    @staticmethod
    def type():
        """
        :return: 返回Reset
        """
        return GateType.Reset

Reset = ResetGate()

class BarrierGate(BasicGate):
    """
    Barrier门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "barrier"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("试图获取Barrier门对应矩阵")

    def __str__(self):
        return "Barrier门"

    @staticmethod
    def type():
        """
        :return: 返回Barrier
        """
        return GateType.Barrier

Barrier = BarrierGate()

class SwapGate(BasicGate):
    """
    Swap门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 2
        self.params = 0
        self.qasm_name = "swap"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,  0,  0,
            0,  0,  1,  0,
            0,  1,  0,  0,
            0,  0,  0,  1
        ], dtype = np.complex)

    def __str__(self):
        return "Swap门"

    @staticmethod
    def type():
        """
        :return: 返回Swap
        """
        return GateType.Swap

    def inverse(self):
        _swap = SwapGate()
        _swap.targs = copy.deepcopy(self.targs)
        return _swap

Swap = SwapGate()

class PermGate(BasicGate):
    """
    Perm门
    """

    @property
    def extra_control_index(self):
        return self.__extra_control_index

    @extra_control_index.setter
    def extra_control_index(self, extra_control_index):
        self.__extra_control_index = extra_control_index

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0
        self.extra_control_index = None

    def __call__(self, other):
        """
        :param other: 置换list或置换tuple
        :raise 类型错误
        :return 修改参数后的self
        """
        self.pargs = []
        if not isinstance(other, list) or not isinstance(other, tuple):
            TypeException("list或tuple", other)
        if isinstance(other, tuple):
            other = list(other)
        length = len(other)
        if length == 0:
            raise Exception("传入的list或tuple不应为空")
        n = int(round(np.log2(length)))
        if (1 << n) != length:
            raise Exception("传入的list或tuple的长度应该为2的幂次")
        self.params = length
        self.targets = n
        for idx in other:
            if not isinstance(idx, int) or idx < 0 or idx >= self.params:
                raise Exception("传入的应该为一个用整数表示的置换数组")
            if idx in self.pargs:
                raise Exception("传入的应该为一个用整数表示的置换数组(不应重复)")
            self.pargs.append(idx)
        self.extra_control_index = None
        return self

    def add_controls_bit(self, index):
        self.extra_control_index = index
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype = np.complex)
        for i in range(self.params):
            for j in range(self.params):
                if self.pargs[i] == j:
                    matrix = np.append(matrix, 1)
                else:
                    matrix = np.append(matrix, 0)
        return matrix

    def __str__(self):
        return "Perm门"

    @staticmethod
    def type():
        """
        :return: 返回Perm
        """
        return GateType.Perm

    def inverse(self):
        _perm = PermGate()
        _perm.targs = copy.deepcopy(self.targs)
        matrix = [0] * self.params
        i = 0
        for parg in self.pargs:
            matrix[parg] = i
            i += 1
        _perm.pargs = matrix
        _perm.params = self.params
        _perm.targets = self.targets
        return _perm

Perm = PermGate()

class ControlPermMulDetailGate(BasicGate):
    """
    Perm门
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 0
        self.params = 0

    def __call__(self, other):
        """
        :param other: 置换list或置换tuple
        :raise 类型错误
        :return 修改参数后的self
        """
        self.pargs = []
        if not isinstance(other, list) or not isinstance(other, tuple):
            TypeException("list或tuple", other)
        if isinstance(other, tuple):
            other = list(other)
        length = len(other)
        if length != 2:
            raise Exception("传入的list或tuple应包含两个值")
        a = other[0]
        N = other[1]
        n = int(np.ceil(np.log2(N)))
        self.params = 2
        self.targets = n
        self.pargs = [a, N]
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype = np.complex)
        a = self.pargs[0]
        N = self.pargs[1]
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx >= N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((idxx * a % N) << 1) + controlxx)
        return matrix

    def __str__(self):
        return "Perm门"

    @staticmethod
    def type():
        """
        :return: 返回Perm
        """
        return GateType.ControlPermMulDetail
ControlPermMulDetail = ControlPermMulDetailGate()

class PermShiftGate(PermGate):
    """
    对一个qureg进行加法或减法操作，本质上是一种PermGate
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """
        :param shift: shift的值
               N:     模数
        :raise: 类型错误
        :return 修改参数后的self
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermShift需要两个参数")
        if not isinstance(N, int):
            raise TypeException("int", N)

        if N <= 0:
            raise Exception("传入的模数应为正整数")
        n = int(round(np.log2(N)))
        if (1 << n) != N:
            raise Exception("传入的N应该为2的幂次")
        self.params = N
        self.targets = n
        for idx in range(N):
            self.pargs.append(((idx + shift) % N + N) % N)
        return self

PermShift = PermShiftGate()

class ControlPermShiftGate(PermGate):
    """
    对一个qureg进行加法或减法操作，本质上是一种PermGate
    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """
        :param shift: shift的值
               N:     模数
        :raise: 类型错误
        :return 修改参数后的self
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermShift需要三个参数")
        if not isinstance(N, int):
            raise TypeException("int", N)

        if N <= 0:
            raise Exception("传入的模数应为正整数")
        n = int(np.ceil(np.log2(N)))
        self.params = N
        self.targets = n + 1
        self.pargs = []
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx < N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((((idxx + shift) % N + N) % N) << 1) + controlxx)
        return self

ControlPermShift = ControlPermShiftGate()

class PermMulGate(PermGate):
    """
    对一个qureg进行乘法操作，本质上是一种PermGate
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """
        :param shift: shift的值
               N:     模数
        :raise: 类型错误
        :return 修改参数后的self
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermMul需要两个参数")
        if not isinstance(N, int):
            raise TypeException("int", N)
        if N <= 0:
            raise Exception("传入的模数应为正整数")
        if shift <= 0:
            raise Exception("传入的乘数应为正整数")

        if np.gcd(shift, N) != 1:
            raise Exception("乘数和模数应当互质,但{}和{}不互质".format(shift, N))

        shift = shift % N

        n = int(round(np.log2(N)))
        if (1 << n) < N:
            n = n + 1
        self.params = N
        self.targets = n
        for idx in range(N):
            self.pargs.append(idx * shift % N)
        for idx in range(N, 1 << n):
            self.pargs.append(idx)
        return self

PermMul = PermMulGate()

class ControlPermMulGate(PermGate):
    """
    对一个qureg进行乘法操作，本质上是一种PermGate
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """
        :param shift: shift的值
               N:     模数
        :raise: 类型错误
        :return 修改参数后的self
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermMul需要两个参数")
        if not isinstance(N, int):
            raise TypeException("int", N)
        if N <= 0:
            raise Exception("传入的模数应为正整数")
        if shift <= 0:
            raise Exception("传入的乘数应为正整数")

        if np.gcd(shift, N) != 1:
            raise Exception("乘数和模数应当互质,但{}和{}不互质".format(shift, N))

        shift = shift % N

        n = int(np.ceil(np.log2(N)))
        self.params = N
        self.targets = n + 1
        self.pargs = []
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx >= N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((idxx * shift % N) << 1) + controlxx)
        return self

ControlPermMul = ControlPermMulGate()

class PermFxGate(PermGate):
    """
    对一个qureg进行f : {0,1}^n -> {0, 1} 的oracle操作，本质上是一种PermGate
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, f):
        """
        :param f:     list
        :raise: 类型错误
        :return 修改参数后的self
        """
        if not isinstance(f, list):
            raise TypeException("list", f)
        n = int(round(np.log2(len(f))))
        if len(f) != 1 << n:
            raise Exception("f的定义域不是{0, 1}^n")
        N = 1 << n
        for i in range(N):
            if f[i] != 0 and f[i] != 1:
                raise Exception("f的值域不是{0, 1}")

        self.params = 1 << (n + 1)
        self.targets = n + 1

        N_2 = N << 1
        for idx in range(N_2):
            if f[idx & (N - 1)] == 1:
                self.pargs.append(idx ^ N)
            else:
                self.pargs.append(idx)
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype = np.complex)
        for i in range(self.params):
            for j in range(self.params):
                if self.pargs[i] == j:
                    np.append(matrix, 1)
                else:
                    np.append(matrix, 0)
        return matrix

    def __str__(self):
        return "Perm门"

    @staticmethod
    def type():
        """
        :return: 返回Perm
        """
        return GateType.Perm
PermFx = PermFxGate()

class CustomGate(BasicGate):
    """
    Custom门
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, other):
        """
        :param other: 置换list或置换tuple
        :raise TypeException
        :return 修改参数后的self
        """
        if not isinstance(other, list) and not isinstance(other, tuple):
            raise TypeException("list或tuple", other)
        if isinstance(other, tuple):
            other = list(other)
        length = len(other)
        if length == 0:
            raise Exception("传入的list或tuple不应为空")
        n2 = int(round(np.sqrt(length)))
        if n2 * n2 != length:
            raise Exception("传入的list或tuple的长度应该为2的幂次的平方")
        n = int(round(np.log2(n2)))
        if (1 << n) != n2:
            raise Exception("传入的list或tuple的长度应该为2的幂次的平方")
        self.targets = n
        self.matrix = np.array([other], dtype = np.complex)
        return self

    def __str__(self):
        return "Custom门"

    @staticmethod
    def type():
        """
        :return: 返回Custom
        """
        return GateType.Custom

    def inverse(self):
        _custom = CustomGate()
        _custom.targs = copy.deepcopy(self.targs)
        _custom.matrix = np.array(np.mat(self.matrix).reshape(1 << self.targets, 1 << self.targets).H.reshape(1, -1),
                                  dtype = np.complex)
        _custom.targets = self.targets
        _custom.params = self.params
        _custom.controls = self.controls
        return _custom

Custom = CustomGate()

class ShorInitalGate(BasicGate):
    """
    Shor IQFT前的初态
    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, other):
        """
        :param other: 置换list或置换tuple
        :raise TypeException
        :return 修改参数后的self
        """
        if not isinstance(other, list) and not isinstance(other, tuple):
            raise TypeException("list或tuple", other)
        if isinstance(other, tuple):
            other = list(other)
        length = len(other)
        if length != 3:
            raise Exception("传入的list或tuple不应包含3个值")
        x = other[0]
        N = other[1]
        u = other[2]
        n = 2 * int(np.ceil(np.log2(N)))
        self.targets = n
        self.pargs = [x, N, u]
        return self

    @staticmethod
    def type():
        """
        :return: 返回Perm
        """
        return GateType.ShorInital

ShorInital = ShorInitalGate()

class GateBuilderModel(object):

    def setGateType(self, type):
        self.gateType = type

    def setPargs(self, pargs):
        """
        :param pargs:
            1) list<int>
            2) int
        :raise TypeException
        """
        if isinstance(pargs, list):
            self.pargs = pargs
        else:
            self.pargs = [pargs]

    def setTargs(self, targs):
        """
        :param targs:
            1) list<int>
            2) int
        """
        if isinstance(targs, list):
            self.targs = targs
        else:
            self.targs = [targs]

    def setCargs(self, cargs):
        """
        :param cargs:
            1) list<int>
            2) int
        """
        if isinstance(cargs, list):
            self.cargs = cargs
        else:
            self.cargs = [cargs]

    def setArgs(self, args):
        """
        :param args:
            1) list<int>
            2) int
        """
        if isinstance(args, list):
            if self.getCargsNumber() > 0:
                self.setCargs(args[0:self.getCargsNumber()])
            if self.getTargsNumber() > 0:
                self.setTargs(args[self.getCargsNumber():])
        else:
            self.setTargs([args])

    def getCargsNumber(self):
        gate = self.__inner_generate_gate__()
        return gate.controls

    def getTargsNumber(self):
        gate = self.__inner_generate_gate__()
        return gate.targets

    def getParamsNumber(self):
        gate = self.__inner_generate_gate__()
        return gate.params

    def getGate(self):
        """
        :raise
            1) 门类型未设置
            2) 参数设置错误
        :return: gate
        """
        gate = self.__inner_generate_gate__()
        return self.__inner_complete_gate__(gate)

    def __inner_generate_gate__(self):
        if self.gateType == GateType.H:
            return HGate()
        elif self.gateType == GateType.S:
            return SGate()
        elif self.gateType == GateType.S_dagger:
            return SDaggerGate()
        elif self.gateType == GateType.X:
            return XGate()
        elif self.gateType == GateType.Y:
            return YGate()
        elif self.gateType == GateType.Z:
            return ZGate()
        elif self.gateType == GateType.ID:
            return IDGate()
        elif self.gateType == GateType.U1:
            return U1Gate()
        elif self.gateType == GateType.U2:
            return U2Gate()
        elif self.gateType == GateType.U3:
            return U3Gate()
        elif self.gateType == GateType.Rx:
            return RxGate()
        elif self.gateType == GateType.Ry:
            return RyGate()
        elif self.gateType == GateType.Rz:
            return RzGate()
        elif self.gateType == GateType.T:
            return TGate()
        elif self.gateType == GateType.T_dagger:
            return TDaggerGate()
        elif self.gateType == GateType.CZ:
            return CZGate()
        elif self.gateType == GateType.CX:
            return CXGate()
        elif self.gateType == GateType.CY:
            return CYGate()
        elif self.gateType == GateType.CH:
            return CHGate()
        elif self.gateType == GateType.CRz:
            return CRzGate()
        elif self.gateType == GateType.CCX:
            return CCXGate()
        elif self.gateType == GateType.Measure:
            return MeasureGate()
        elif self.gateType == GateType.Swap:
            return SwapGate()
        elif self.gateType == GateType.Perm:
            return PermGate()
        elif self.gateType == GateType.Custom:
            return CustomGate()
        elif self.gateType == GateType.Reset:
            return ResetGate()
        raise Exception("未设置门类型或不支持的门类型")

    def __inner_complete_gate__(self, gate : BasicGate):
        """
        :raise 参数设置错误
        :return: gate
        """
        if self.gateType == GateType.Perm:
            gate = gate(self.pargs)
        elif self.gateType == GateType.Custom:
            gate = gate(self.pargs)
        if gate.targets != 0:
            if len(self.targs) == gate.targets:
                gate.targs = copy.deepcopy(self.targs)
            else:
                raise Exception("作用位数量错误")

        if gate.controls != 0:
            if len(self.cargs) == gate.controls:
                gate.cargs = copy.deepcopy(self.cargs)
            else:
                raise Exception("控制位数量错误")
        if gate.params != 0 and self.gateType != GateType.Perm:
            if len(self.pargs) == gate.params:
               gate.pargs = copy.deepcopy(self.pargs)
            else:
                raise Exception("参数位数量错误")

        return gate

    '''
    @staticmethod
    def refine_gate(circuit : Circuit, index = None):
        if index is None:
            index = [i for i in range(len(circuit.qubits))]
        gates = []
        builder = GateBuilderModel()
        for gate in circuit.gates:
            builder.setGateType(gate.type)
            builder.setCargs(gate.cargs)
            builder.setTargs(gate.targs)
            builder.setPargs(gate.pargs)
            gates.append(builder.getGate())
        return gates

    def __init__(self):
        self.gateType = GateType.Error
        self.pargs = []
        self.targs = []
        self.cargs = []
    '''
    @staticmethod
    def reflect_gates(gates : list):
        reflect = []
        l_g = len(gates)
        for index in range(l_g - 1, -1, -1):
            reflect.append(gates[index].inverse())
        return reflect

    @staticmethod
    def reflect_apply_gates(gates: list, circuit):
        l_g = len(gates)
        for index in range(l_g - 1, -1, -1):
            gate = gates[index].inverse()
            qubits = []
            for control in gate.cargs:
                qubits.append(circuit[control])
            for target in gate.targs:
                qubits.append(circuit[target])
            circuit.__add_qureg_gate__(gate, qubits)

GateBuilder = GateBuilderModel()

class ExtensionGateType(Enum):
    QFT = 0
    IQFT = 1
    RZZ = 2
    CU1 = 3
    CU3 = 4
    Fredkin = 5
    CCX = 6
    CRz = 7
    CCRz = 8

class gateModel(object):
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

        gates = self.build_gate(len(qureg))
        if isinstance(gates, Circuit):
            gates = gates.gates
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

    def build_gate(self, other):
        GateBuilder.setGateType(GateType.ID)
        GateBuilder.setTargs(other)
        return [GateBuilder.getGate()]

class QFTModel(gateModel):
    def __or__(self, other):
        """
        给上QFT对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        for i in range(len(other)):
            H | qureg[i]
            for j in range(i + 1, len(other)):
                CRz(2 * np.pi / (1 << j - i + 1)) | (qureg[j], qureg[i])

    def build_gate(self, other):
        gates = []
        for i in range(len(other)):
            GateBuilder.setGateType(GateType.H)
            GateBuilder.setTargs(other[i])
            gates.append(GateBuilder.getGate())

            GateBuilder.setGateType(GateType.CRz)
            for j in range(i + 1, len(other)):
                GateBuilder.setPargs(2 * np.pi / (1 << j - i + 1))
                GateBuilder.setCargs(other[j])
                GateBuilder.setTargs(other[i])
                gates.append(GateBuilder.getGate())
        return gates

QFT = QFTModel()

class IQFTModel(gateModel):
    def __or__(self, other):
        """
        给上QFT对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        for i in range(len(other) - 1, -1, -1):
            for j in range(len(other) - 1, i, -1):
                CRz(-2 * np.pi / (1 << j - i + 1)) | (qureg[j], qureg[i])
            H | qureg[i]

    def build_gate(self, other):
        gates = []
        for i in range(len(other) - 1, -1, -1):
            GateBuilder.setGateType(GateType.CRz)
            for j in range(len(other) - 1, i, -1):
                GateBuilder.setPargs(-2 * np.pi / (1 << j - i + 1))
                GateBuilder.setCargs(other[j])
                GateBuilder.setTargs(other[i])
                gates.append(GateBuilder.getGate())
            GateBuilder.setGateType(GateType.H)
            GateBuilder.setTargs(other[i])
            gates.append(GateBuilder.getGate())
        return gates

IQFT = IQFTModel()

class RZZModel(gateModel):
    def __or__(self, other):
        """
        给上RZZ对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        CX | (qureg[0], qureg[1])
        U1(self.parg) | qureg[1]
        CX | (qureg[0], qureg[1])

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(self.parg)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

RZZ = RZZModel()

class CU1Gate(gateModel):
    def __or__(self, other):
        """
        给上CU1对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        U1(self.parg / 2) | qureg[0]
        CX | (qureg[0], qureg[1])
        U1(-self.parg / 2) | qureg[1]
        CX | (qureg[0], qureg[1])
        U1(self.parg / 2) | qureg[1]

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(-self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CU1 = CU1Gate()

class CRz_DecomposeModel(gateModel):
    def __or__(self, other):
        """
        给上CRz对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        Rz(self.parg / 2) | qureg[0]
        CX | (qureg[0], qureg[1])
        Rz(-self.parg / 2) | qureg[1]
        CX | (qureg[0], qureg[1])
        Rz(self.parg / 2) | qureg[1]

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(-self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CRz_Decompose = CRz_DecomposeModel()

class CU3Gate(gateModel):
    def __or__(self, other):
        """
        给上CU3对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        U1((self.pargs[1] + self.pargs[2]) / 2) | qureg[0]
        U1(self.pargs[2] - self.pargs[1]) | (qureg[1])
        CX | (qureg[0], qureg[1])
        U3([-self.pargs[0] / 2, 0, -(self.pargs[0] + self.pargs[1]) / 2]) | qureg[1]
        CX | (qureg[0], qureg[1])
        U3([self.pargs[0] / 2, self.pargs[1], 0]) | qureg[1]

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs((self.pargs[1] + self.pargs[2]) / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setPargs(self.pargs[2] + self.pargs[1])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U3)
        GateBuilder.setPargs([-self.pargs[0] / 2, 0, -(self.pargs[0] + self.pargs[1]) / 2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U3)
        GateBuilder.setPargs([self.pargs[0] / 2, self.pargs[1], 0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CU3 = CU3Gate()

class CCRzModel(gateModel):
    def __or__(self, other):
        """
        给上CCRz对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        CRz_Decompose(self.parg / 2)  | (qureg[1], qureg[2])
        CX                            | (qureg[0], qureg[1])
        CRz_Decompose(-self.parg / 2) | (qureg[1], qureg[2])
        CX                            | (qureg[0], qureg[1])
        CRz_Decompose(self.parg / 2)  | (qureg[0], qureg[2])

    def build_gate(self, other):
        gates = CRz_Decompose(self.parg / 2).build_gate(other)

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        gates = CRz_Decompose(-self.parg / 2).build_gate(other)

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        gates = CRz_Decompose(self.parg / 2).build_gate(other)

        return gates

CCRz = CCRzModel()

class FredkinModel(gateModel):
    def __or__(self, other):
        """
        给上Fredkin对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        CX            | (qureg[2], qureg[1])
        CCX_Decompose | (qureg[0], qureg[1], qureg[2])
        CX            | (qureg[2], qureg[1])

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        gates.extend(CCX_Decompose.build_gate(other))

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

Fredkin = FredkinModel()

class CCX_DecomposeModel(gateModel):
    def __or__(self, other):
        """
        给上Fredkin对应的门
        :param other 转换的对象
                1）tuple<qubit, qureg>
                2) qureg/list<qubit, qureg>
                3) Circuit
        :raise TypeException
        """
        qureg = self.qureg_trans(other)
        H           | qureg[2]
        CX          | (qureg[1], qureg[2])
        T_dagger    | qureg[2]
        CX          | (qureg[0], qureg[2])
        T           | qureg[2]
        CX          | (qureg[1], qureg[2])
        T_dagger    | qureg[2]
        CX          | (qureg[0], qureg[2])
        T           | qureg[1]
        T           | qureg[2]
        H           | qureg[2]
        CX          | (qureg[0], qureg[1])
        T           | qureg[0]
        T_dagger    | qureg[1]
        CX          | (qureg[0], qureg[1])

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.H)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[1])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T_dagger)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[1])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T_dagger)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.H)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T_dagger)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CCX_Decompose = CCX_DecomposeModel()

class ExtensionGateBuilderModel(object):

    def setGateType(self, type):
        self.gateType = type

    def setPargs(self, pargs):
        """
        :param pargs:
            1) list<int>
            2) int
        :raise TypeException
        """
        if isinstance(pargs, list):
            self.pargs = pargs
        else:
            self.pargs = [pargs]

    def setTargs(self, targs):
        """
        :param targs:
            1) list<int>
            2) int
        """
        if isinstance(targs, list):
            self.targs = targs
        else:
            self.targs = [targs]

    def getTargsNumber(self):
        gate = self.__inner_generate_gate__()
        return gate.targets

    def getParamsNumber(self):
        gate = self.__inner_generate_gate__()
        return gate.params

    def getGate(self):
        """
        :raise
            1) 门类型未设置
            2) 参数设置错误
        :return: gate
        """
        return self.__inner_generate_gate__()

    def __inner_generate_gate__(self):
        if self.gateType == ExtensionGateType.QFT:
            return QFT.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.IQFT:
            return IQFT.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.RZZ:
            return RZZ(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CU1:
            return CU1(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CU3:
            return CU3(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.Fredkin:
            return Fredkin.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CCX:
            return CCX_Decompose.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CRz:
            return CRz_Decompose(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CCRz:
            return CCRz(self.pargs).build_gate(self.targs)

        raise Exception("未设置门类型或不支持的门类型")

    def __init__(self):
        self.gateType = GateType.Error
        self.pargs = []
        self.targs = []
        self.cargs = []

ExtensionGateBuilder = ExtensionGateBuilderModel()

class GateDigitException(Exception):
    def __init__(self, controls, targets, indeed):
        """
        :param controls: 控制位数量
        :param targets: 作用位数量
        :param indeed: 实际传入
        """
        Exception.__init__(self, "类型错误,应传入{}个控制位,{}个作用位,/"
                                 "共{}位,实际传入了{}个数".format(controls, targets, controls + targets, indeed))
