#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 8:44 下午
# @Author  : Han Yu
# @File    : _qubit.py

from QuICT.exception import TypeException, FrameworkException, CircuitStructException
from QuICT.backends import systemCdll
import numpy as np
import random
from ctypes import *
from math import sqrt
import sys
import weakref

qubit_id = 0
tangle_id = 0

class Qubit(object):
    """
    类的属性
    """

    # qubit的id
    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    # qubit所属的tangle
    @property
    def tangle(self):
        if self.__tangle is None:
            self.__tangle = Tangle(self)
        return self.__tangle

    @tangle.setter
    def tangle(self, tangle):
        self.__tangle = tangle

    # 所属电路
    @property
    def circuit(self):
        return self.__circuit()

    @circuit.setter
    def circuit(self, circuit):
        self.__circuit = weakref.ref(circuit)

    # 测试后值
    @property
    def measured(self) -> int:
        return self.__measured

    @measured.setter
    def measured(self, measured):
        self.__measured = measured

    # 概率
    @property
    def prop(self) -> int:
        return self.__prop

    @prop.setter
    def prop(self, prop):
        self.__prop = prop

    """
    基础方法改写
    """
    def __init__(self, circuit):
        global qubit_id
        self.__id = qubit_id
        qubit_id = qubit_id + 1
        self.circuit = circuit
        self.__tangle = None
        self.__measured = -1
        self.__prop = 0.0

    def __str__(self):
        """
        :return: 所属电路与自身id
        """
        return str("电路 " + self.circuit + " 中的" + self.id)

    def __int__(self):
        """
        输出测量值
        :return: 测量值
        :raise 没有在该位上添加过测量门
        """
        if self.measured == -1:
            raise Exception("该位'{}'没有被测量过".format(self.id))
        return self.measured

    def __bool__(self):
        """
        测量后值的布尔意义
        :return: 布尔值
        :raise 没有在该位上添加过测量门
        """
        if self.measured == -1:
            raise Exception("该位'{}'没有被测量过".format(self.id))
        if self.measured == 0:
            return False
        else:
            return True

    def has_tangle(self):
        if self.__tangle is None:
            return False
        else:
            return True

    def tangle_clear(self):
        if self.__tangle is not None:
            del self.__tangle
        self.__tangle = None

class Qureg(list):
    # 所属电路
    @property
    def circuit(self):
        if len(self) == 0:
            raise Exception("非法操作，该Qureg为空")
        return self[0].circuit

    @circuit.setter
    def circuit(self, circuit):
        if len(self) == 0:
            raise Exception("非法操作，该Qureg为空")
        for qubit in self:
            qubit.circuit = circuit

    def __init__(self, other = None):
        """
        初始化
        :param other:
            1) Qubit/tuple<Qubit>
            2) Qureg
            3) Circuit
        """
        from ._circuit import Circuit
        super().__init__()
        if other is None:
            return
        if isinstance(other, Qubit):
            self.append(other)
        elif isinstance(other, tuple):
            for qubit in other:
                if not isinstance(qubit, Qubit):
                    raise TypeException("Qubit/tuple<Qubit>或Qureg或Circuit", other)
                self.append(qubit)
        elif isinstance(other, Qureg):
            for qubit in other:
                self.append(qubit)
        elif isinstance(other, Circuit):
            for qubit in other.qubits:
                self.append(qubit)
        else:
            raise TypeException("Qubit/tuple<Qubit>或Qureg或Circuit", other)

    def __str__(self):
        return str(self.__int__())

    def __int__(self):
        for qubit in self:
            if qubit.measured == -1:
                string = "该位:"+str(qubit.id)+"没有被测量过"
                raise Exception(string)
        value = 0
        for i in range(len(self)):
            value <<= 1
            if self[i].measured == 1:
                value += 1
        return value

    def force_assign_random(self):
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_tangle():
                raise Exception("某qubit已有初值")
        tangle = Tangle(self[0])
        for i in range(1, len(self)):
            tangle.qureg.append(self[i])
        tangle.force_assign_random()
        for qubit in self:
            qubit.tangle = tangle

    def force_assign_zeros(self):
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_tangle():
                raise Exception("某qubit已有初值")
        tangle = Tangle(self[0])
        for i in range(1, len(self)):
            tangle.qureg.append(self[i])
        tangle.force_assign_zeros()
        for qubit in self:
            qubit.tangle = tangle

    def force_copy(self, other, copy_list):
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_tangle():
                raise Exception("某qubit已有初值")
        tangle = Tangle(self[copy_list[0]])
        self[copy_list[0]].tangle = tangle
        for i in range(1, len(copy_list)):
            tangle.qureg.append(self[copy_list[i]])
            self[copy_list[i]].tangle = tangle
        tangle.force_copy(other)

class Tangle(object):
    """
    计算纠缠块
    """
    """
    类的属性
    """

    # tangle的id
    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    # tangle包含的qureg
    @property
    def qureg(self) -> Qureg:
        return self.__qureg

    @qureg.setter
    def qureg(self, qureg):
        self.__qureg = qureg

    # tangle包含的值
    @property
    def values(self) -> np.array:
        return self.__values

    @values.setter
    def values(self, values):
        self.__values = values

    def __init__(self, qubit):
        self.__qureg = Qureg(qubit)
        if qubit.measured == -1 or qubit.measured == 0:
            self.__values = np.array([1, 0], dtype=np.complex)
        else:
            self.__values = np.array([0, 1], dtype=np.complex)
        global tangle_id
        self.__id = tangle_id
        tangle_id = tangle_id + 1

    def print_infomation(self):
        print(self.values, self.id, len(self.qureg))

    def index_for_qubit(self, qubit) -> int:
        """
        :param qubit: 需要查询的qubit
        :return: 索引值
        :raise 传入的参数不是qubit，或者不在该tangle中
        """
        if not isinstance(qubit, Qubit):
            raise TypeException("Qubit", qubit)
        for i in range(len(self.qureg)):
            if self.qureg[i].id == qubit.id:
                return i
        raise Exception("传入的qubit不在该Tangle中")

    def merge(self, other):
        """
        :param other: 需要合并的Tangle
        """
        if self.id == other.id:
            return
        if len(set(self.qureg).intersection(set(other.qureg))) != 0:
            return

        dll = systemCdll.quick_operator_cdll
        merge_operator_func = dll.merge_operator_func
        merge_operator_func.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]
        length = (1 << len(self.qureg)) * (1 << len(other.qureg))
        # merge_operator_func.restype = np.ctypeslib.ndpointer(dtype = np.complex, shape=(length, ))
        merge_operator_func.restype = None
        values = np.zeros(length, dtype = np.complex)
        merge_operator_func(
            len(self.qureg),
            self.values,
            len(other.qureg),
            other.values,
            values
        )
        # self.values = np.ctypeslib.as_array(ndpointer, shape=(length, ))
        for qubit in other.qureg:
            qubit.tangle = self
        self.qureg.extend(other.qureg)
        del other
        self.values = values

    def deal_single_gate(self, other, has_fidelity = False, fidelity = 1.0):
        """
        :param other: 需要作用的门
               has_fidelity: 有保真度调节
               fidelity: 保真度
        :raise 索引错误, 电路错误
        """
        dll = systemCdll.quick_operator_cdll
        single_operator_func = dll.single_operator_func
        single_operator_func.argtypes = [
            c_int,
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]

        index = 0
        qubit = self.qureg.circuit.qubits[other.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            index = index + 1
        if index == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")

        matrix = other.matrix
        if has_fidelity:
            theta = np.arccos(fidelity / np.sqrt(2)) - np.pi / 4
            theta *= (random.random() - 0.5) * 2
            RyMatrix = np.array(
                [
                    np.cos(theta), -np.sin(theta),
                    np.sin(theta), np.cos(theta)
                ]
                , dtype=np.complex
            )
            # print(fidelity, theta, RyMatrix)
            # matrix = RyMatrix * matrix
            Ry0 = RyMatrix[0] * matrix[0] + RyMatrix[1] * matrix[2]
            Ry1 = RyMatrix[0] * matrix[1] + RyMatrix[1] * matrix[3]
            Ry2 = RyMatrix[2] * matrix[0] + RyMatrix[3] * matrix[2]
            Ry3 = RyMatrix[2] * matrix[1] + RyMatrix[3] * matrix[3]
            matrix[0] = Ry0
            matrix[1] = Ry1
            matrix[2] = Ry2
            matrix[3] = Ry3
        single_operator_func(
            len(self.qureg),
            index,
            self.values,
            matrix
        )

    def deal_measure_gate(self, other):
        """
        :param other: 需要作用的测量门
        :raise 索引错误
        """
        dll = systemCdll.quick_operator_cdll
        measure_operator_func = dll.measure_operator_func
        measure_operator_func.argtypes = [
            c_int,
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            c_double,
            POINTER(c_double)
        ]
        measure_operator_func.restype = c_bool

        index = 0
        qubit = self.qureg.circuit.qubits[other.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            index = index + 1
        if index == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")
        generation = random.random()
        # print(generation)

        prop = c_double()
        result = measure_operator_func(
            len(self.qureg),
            index,
            self.values,
            generation,
            pointer(prop)
        )
        self.qureg.remove(qubit)
        self.values = self.values[:(1 << len(self.qureg))]
        qubit.tangle = None
        qubit.measured = result
        qubit.prop = prop.value

    def deal_reset_gate(self, other):
        """
        :param other: 需要作用的测量门
        :raise 索引错误
        """
        dll = systemCdll.quick_operator_cdll
        reset_operator_func = dll.reset_operator_func
        reset_operator_func.argtypes = [
            c_int,
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]

        index = 0
        qubit = self.qureg.circuit.qubits[other.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            index = index + 1
        if index == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")
        reset_operator_func(
            len(self.qureg),
            index,
            self.values
        )
        self.qureg.remove(qubit)
        self.values = self.values[:(1 << len(self.qureg))]
        qubit.tangle = None

    def deal_control_single_gate(self, other):
        """
        :param other: 需要作用的门
        :raise 索引错误
        """
        dll = systemCdll.quick_operator_cdll
        control_single_operator_func = dll.control_single_operator_func
        control_single_operator_func.argtypes = [
            c_int,
            c_int,
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]
        cindex = 0
        qubit = self.qureg.circuit.qubits[other.carg]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex = cindex + 1
        if cindex == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")

        tindex = 0
        qubit = self.qureg.circuit.qubits[other.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            tindex = tindex + 1
        if tindex == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")
        # print("before:", np.round(self.values, decimals = 2))
        # print(cindex, tindex)
        control_single_operator_func(
            len(self.qureg),
            cindex,
            tindex,
            self.values,
            other.matrix
        )
        # print("after:", np.round(self.values, decimals=2))

    def deal_ccx_gate(self, other):
        """
        :param other: 需要作用的CCX门
        :raise 索引错误
        """
        dll = systemCdll.quick_operator_cdll
        ccx_single_operator_func = dll.ccx_single_operator_func
        ccx_single_operator_func.argtypes = [
            c_int,
            c_int,
            c_int,
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]
        cindex1 = 0
        qubit = self.qureg.circuit.qubits[other.cargs[0]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex1 = cindex1 + 1
        if cindex1 == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")

        cindex2 = 0
        qubit = self.qureg.circuit.qubits[other.cargs[1]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex2 = cindex2 + 1
        if cindex2 == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")

        tindex = 0
        qubit = self.qureg.circuit.qubits[other.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            tindex = tindex + 1
        if tindex == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")
        ccx_single_operator_func(
            len(self.qureg),
            cindex1,
            cindex2,
            tindex,
            self.values
        )

    def deal_swap_gate(self, other):
        """
        :param other: 需要处理的swap门
        """
        cindex = 0
        qubit = self.qureg.circuit.qubits[other.targs[0]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex = cindex + 1
        if cindex == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")

        tindex = 0
        qubit = self.qureg.circuit.qubits[other.targs[1]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            tindex = tindex + 1
        if tindex == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")

        t = self.qureg[cindex]
        self.qureg[cindex] = self.qureg[tindex]
        self.qureg[tindex] = t

    def deal_custom_gate(self, other):
        """
        :param other: 作用的任意酉矩阵
        """

        dll = systemCdll.quick_operator_cdll
        custom_operator_gate = dll.custom_operator_gate
        custom_operator_gate.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            POINTER(c_int),
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]

        index = np.array([])
        for idx in other.targs:
            qubit = self.qureg.circuit.qubits[idx]
            temp_idx = 0
            for test in self.qureg:
                if test.id == qubit.id:
                    break
                temp_idx = temp_idx + 1
            if temp_idx == len(self.qureg):
                raise FrameworkException("索引不在对应纠缠块中")
            np.append(index, temp_idx)

        custom_operator_gate(
            len(self.qureg),
            self.values,
            index,
            other.targets,
            other.matrix
        )

    def deal_perm_gate(self, other):
        """
        :param other: 待处理的置换门
        """
        dll = systemCdll.quick_operator_cdll
        perm_operator_gate = dll.perm_operator_gate
        perm_operator_gate.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
            c_int,
            np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS")
        ]

        index = np.array([], dtype=np.int)
        targs = other.targs
        if not isinstance(targs, list):
            targs = [targs]
        for idx in targs:
            qubit = self.qureg.circuit.qubits[idx]
            temp_idx = 0
            for test in self.qureg:
                if test.id == qubit.id:
                    break
                temp_idx = temp_idx + 1
            if temp_idx == len(self.qureg):
                raise FrameworkException("索引不在对应纠缠块中")
            index = np.append(index, temp_idx)
        perm_operator_gate(
            len(self.qureg),
            self.values,
            index,
            other.targets,
            np.array(other.pargs, dtype=np.int)
        )

    def deal_controlMulPerm_gate(self, other):
        """
        :param other: 待处理的置换门
        """
        dll = systemCdll.quick_operator_cdll
        perm_operator_gate = dll.control_mul_perm_operator_gate
        perm_operator_gate.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            c_int,
            c_int
        ]

        index = np.array([], dtype=np.int)
        targs = other.targs
        if not isinstance(targs, list):
            targs = [targs]
        for idx in targs:
            qubit = self.qureg.circuit.qubits[idx]
            temp_idx = 0
            for test in self.qureg:
                if test.id == qubit.id:
                    break
                temp_idx = temp_idx + 1
            if temp_idx == len(self.qureg):
                raise FrameworkException("索引不在对应纠缠块中")
            index = np.append(index, temp_idx)
        control = other.cargs[0]
        qubit = self.qureg.circuit.qubits[control]
        temp_idx = 0
        for test in self.qureg:
            if test.id == qubit.id:
                break
            temp_idx = temp_idx + 1
        if temp_idx == len(self.qureg):
            raise FrameworkException("索引不在对应纠缠块中")
        control = temp_idx
        perm_operator_gate(
            len(self.qureg),
            self.values,
            index,
            control,
            other.targets,
            other.pargs[0],
            other.pargs[1]
        )

    def deal_shorInitial_gate(self, other):
        """
        :param other: 待处理的置换门
        """
        dll = systemCdll.quick_operator_cdll
        perm_operator_gate = dll.shor_classical_initial_gate
        perm_operator_gate.argtypes = [
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            c_int,
            c_int
        ]

        index = np.array([], dtype=np.int)
        targs = other.targs
        if not isinstance(targs, list):
            targs = [targs]
        for idx in targs:
            qubit = self.qureg.circuit.qubits[idx]
            temp_idx = 0
            for test in self.qureg:
                if test.id == qubit.id:
                    break
                temp_idx = temp_idx + 1
            if temp_idx == len(self.qureg):
                raise FrameworkException("索引不在对应纠缠块中")
            index = np.append(index, temp_idx)
        perm_operator_gate(
            len(self.qureg),
            self.values,
            index,
            other.targets,
            other.pargs[0],
            other.pargs[1],
            other.pargs[2]
        )

    def force_assign_random(self):
        self.values = np.zeros(1 << len(self.qureg), dtype=np.complex)
        sqrnorm = 0
        for i in range(1 << len(self.qureg)):
            # if i == 0:
            #    self.values[i] = 1
            # else:
            #    self.values[i] = 0
            real = random.random()
            imag = random.random()
            self.values[i] = real + imag * 1j
            norm = abs(self.values[i])
            sqrnorm += norm * norm
        sqrnorm = sqrt(sqrnorm)
        for i in range(1 << len(self.qureg)):
            self.values[i] /= sqrnorm

    def force_assign_zeros(self):
        self.values = np.zeros(1 << len(self.qureg), dtype=np.complex)
        self.values[0] = 1 + 0j

    def force_copy(self, other):
        self.values = other.values.copy()

    def partial_prop(self, other):
        back = [0.0] * (1 << len(other))
        save_list = []
        for id in other:
            for index in range(len(self.qureg)):
                if self.qureg[index].id == id:
                    save_list.append(index)
        lv = len(self.values)
        lq = len(self.qureg)
        lo = len(other)
        for i in range(lv):
            pos = 0
            for j in range(len(save_list)):
                if (i & (1 << (lq - 1 - save_list[j]))) != 0:
                    pos += (1 << j)
            norm = abs(self.values[i])
            back[pos] += norm * norm
        return back

    def __del__(self):
        self.values = None
        self.qureg = None
