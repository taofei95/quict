#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 8:44 下午
# @Author  : Han Yu
# @File    : _qubit.py

from collections.abc import Iterable
from ctypes import *
from math import sqrt
import random
import weakref

import numpy as np

from QuICT.exception import FrameworkException, \
    IndexDuplicateException, IndexLimitException, TypeException
from QuICT.backends import systemCdll

# global qubit id count
qubit_id = 0

# global tangle id count
tangle_id = 0

class Qubit(object):
    """ Implement a quantum bit

    Qubit is the basic unit of quantum circuits, it will appear with some certain circuit.

    Attributes:
        id(int): the unique identity code of a qubit, which is generated globally.
        tangle(Tangle):
            a special qureg in which all qubits may entangle with each other.
            class Tangle will be defined and introduced below.
        circuit(Circuit): the circuit this qubit belonged to.
        measured(int):
            the measure result of the qubit.
            After apply measure gate on the qubit, the measured will be 0 or 1,
            otherwise raise an exception
        prob(float):
            the probability of measure result to be 1, which range in [0, 1].
            After apply measure gate on the qubit, this attribute can be read,
            otherwise raise an exception
    """

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
    def prob(self) -> float:
        return self.__prob

    @prob.setter
    def prob(self, prob):
        self.__prob = prob

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
        self.__prob = 0.0

    def __str__(self):
        """ string describe of the qubit

        Returns:
            str: a simple describe
        """
        return f"电路 {self.circuit} 中的{self.id}"

    def __int__(self):
        """ int value of the qubit(measure result)

        Returns:
            int: measure result

        Raises:
            The qubit has not be measured.
        """
        if self.measured == -1:
            raise Exception(f"The qubit {self.id} has not be measured")
        return self.measured

    def __bool__(self):
        """ int value of the qubit(measure result)

        Returns:
            bool: measure result

        Raises:
            The qubit has not be measured.
        """
        if self.measured == -1:
            raise Exception(f"The qubit {self.id} has not be measured")
        if self.measured == 0:
            return False
        else:
            return True

    def has_tangle(self):
        """ whether the qubit is belonged to a tangle.

        Returns:
            bool: True if qubit is belonged to a tangle
        """
        if self.__tangle is None:
            return False
        else:
            return True

    def tangle_clear(self):
        """ delete the tangle of the qubit

        """
        if self.__tangle is not None:
            del self.__tangle
        self.__tangle = None

class Qureg(list):
    """ Implement a quantum register

    Qureg is a list of Qubit, which is a subClass of list.

    Attributes:
        circuit(Circuit): the circuit this qureg belonged to.
    """

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

    def __init__(self, qubits = None):
        """ initial a qureg with qubit(s)

        Args:
            qubits: the qubits which make up the qureg, it can have below form,
                1) Circuit
                2) Qureg
                3) Qubit
                4) tuple<Qubit>
        """
        from ._circuit import Circuit
        super().__init__()
        if qubits is None:
            return
        if isinstance(qubits, Qubit):
            self.append(qubits)
        elif isinstance(qubits, tuple):
            for qubit in qubits:
                if not isinstance(qubit, Qubit):
                    raise TypeException("Qubit/tuple<Qubit> or Qureg or Circuit", qubits)
                self.append(qubit)
        elif isinstance(qubits, Qureg):
            for qubit in qubits:
                self.append(qubit)
        elif isinstance(qubits, Circuit):
            for qubit in qubits.qubits:
                self.append(qubit)
        else:
            raise TypeException("Qubit/tuple<Qubit> or Qureg or Circuit", qubits)

    def __call__(self, indexes: object):
        """ get a smaller qureg from this qureg

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) list<int>
                3) tuple<int>
        Returns:
            Qureg: the qureg correspond to the indexes
        Exceptions:
            IndexDuplicateException: the range of indexes is error.
            TypeException: the type of indexes is error.
        """

        # int
        if isinstance(indexes, int):
            if indexes < 0 or indexes >= len(self):
                raise IndexLimitException(len(self), indexes)
            return Qureg(self[indexes])

        # tuple
        if isinstance(indexes, tuple):
            indexes = list(indexes)

        # list
        if isinstance(indexes, list):
            if len(indexes) != len(set(indexes)):
                raise IndexDuplicateException(indexes)
            qureg = Qureg()
            for element in indexes:
                if not isinstance(element, int):
                    raise TypeException("int", element)
                if element < 0 or element >= len(self):
                    raise IndexLimitException(len(self), element)
                qureg.append(self[element])
            return qureg

        raise TypeException("int or list or tuple", indexes)

    def __int__(self):
        """ the value of the register

        Return the value of the register if all qubits have been measured.
        Note that the compute mode is BigEndian.

        Returns:
            int: the value of the register

        Raises:
            Exception: some qubit has not be measured
        """
        for qubit in self:
            if qubit.measured == -1:
                raise Exception(f"The qubit {qubit.id} has not be measured")
        value = 0
        for i in range(len(self)):
            value <<= 1
            if self[i].measured == 1:
                value += 1
        return value

    def __str__(self):
        """ a simple describe

        Returns:
            str: the value of the qureg

        """
        return str(self.__int__())

    def __getitem__(self, item):
        """ to fit the slice operator, overloaded this function.

        get a smaller qureg/qubit from this qureg

        Args:
            item(int/slice): slice passed in.
        Return:
            Qubit/Qureg: the result or slice
        """
        if isinstance(item, int):
            return super().__getitem__(item)
        elif isinstance(item, slice):
            qureg_list = super().__getitem__(item)
            qureg = Qureg()
            for qubit in qureg_list:
                qureg.append(qubit)
            return qureg

    def force_assign_random(self):
        """ assign random values for qureg which has initial values

        after calling this function, all qubits will be in a Tangle.

        """
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_tangle():
                raise Exception(f"qubit {qubit.id} has initial value")
        tangle = Tangle(self[0])
        for i in range(1, len(self)):
            tangle.qureg.append(self[i])
        tangle.force_assign_random()
        for qubit in self:
            qubit.tangle = tangle

    def force_assign_zeros(self):
        """ assign zero for qureg has initial values

        after calling this function, all qubits will be in a Tangle.

        """
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_tangle():
                raise Exception(f"qubit {qubit.id} has initial value")
        tangle = Tangle(self[0])
        for i in range(1, len(self)):
            tangle.qureg.append(self[i])
        tangle.force_assign_zeros()
        for qubit in self:
            qubit.tangle = tangle

    def force_copy(self, copy_item, indexes):
        """ copy values from other Tangle for this qureg which has initial values

        after calling this function, all qubits will be in a Tangle.

        Args:
            copy_item(Tangle): the Tangle need to be copied.
            indexes(list<int>): the indexes of goal qubits

        """
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_tangle():
                raise Exception(f"qubit {qubit.id} has initial value")
        tangle = Tangle(self[indexes[0]])
        self[indexes[0]].tangle = tangle
        for i in range(1, len(indexes)):
            tangle.qureg.append(self[indexes[i]])
            self[indexes[i]].tangle = tangle
        tangle.force_copy(copy_item)

class Tangle(object):
    """ Implement a tangle

    a basic computation unit of the amplitude simulation, in which
    all qubits are seemed entangle with each other.

    Attributes:
        id(int): the unique identity code of a tangle, which is generated globally.
        qureg(Qureg): the qubit register of the tangle
        values(np.array): the inner amplitude of this tangle

    """

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def qureg(self) -> Qureg:
        return self.__qureg

    @qureg.setter
    def qureg(self, qureg):
        self.__qureg = qureg

    @property
    def values(self) -> np.array:
        return self.__values

    @values.setter
    def values(self, values):
        self.__values = values

    # life cycle
    def __init__(self, qubit):
        """ initial a tangle with one qubit

        Args:
            qubit: the qubit form the tangle.

        """
        self.__qureg = Qureg(qubit)
        if qubit.measured == -1 or qubit.measured == 0:
            self.__values = np.array([1, 0], dtype=np.complex)
        else:
            self.__values = np.array([0, 1], dtype=np.complex)
        global tangle_id
        self.__id = tangle_id
        tangle_id = tangle_id + 1

    def __del__(self):
        """ release the memory

        """
        self.values = None
        self.qureg = None

    def print_infomation(self):
        """ print the infomation of the tangle

        """
        print(self.values, self.id, len(self.qureg))

    def index_for_qubit(self, qubit) -> int:
        """ find the index of qubit in this tangle's qureg

        Args:
            qubit(Qubit): the qubit need to be indexed.

        Returns:
            int: the index of the qubit.

        Raises:
            Exception: the qubit is not in the tangle
        """
        if not isinstance(qubit, Qubit):
            raise TypeException("Qubit", qubit)
        for i in range(len(self.qureg)):
            if self.qureg[i].id == qubit.id:
                return i
        raise Exception("the qubit is not in the tangle")

    def merge(self, other):
        """ merge another tangle into this tangle

        Args:
            other: the tangle need to be merged.

        Exceptions:
            FrameworkException: the index is out of range
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

    def deal_single_gate(self, gate, has_fidelity = False, fidelity = 1.0):
        """ apply an one-qubit gate on this tangle

        Args:
            gate(BasicGate): the gate to be applied.
            has_fidelity(bool): whether gate is completely accurate.
            fidelity(float): the fidelity of the gate

        Exceptions:
            FrameworkException: the index is out of range
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
        qubit = self.qureg.circuit.qubits[gate.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            index = index + 1
        if index == len(self.qureg):
            raise FrameworkException("the index is out of range")

        matrix = gate.matrix
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

    def deal_measure_gate(self, gate):
        """ apply a measure gate on this tangle

        Note that after flush the measure gate, the qubit will be removed
        from the tangle.

        Args:
            gate(MeasureGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        qubit = self.qureg.circuit.qubits[gate.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            index = index + 1
        if index == len(self.qureg):
            raise FrameworkException("the index is out of range")
        generation = random.random()
        # print(generation)

        prob = c_double()
        result = measure_operator_func(
            len(self.qureg),
            index,
            self.values,
            generation,
            pointer(prob)
        )
        self.qureg.remove(qubit)
        self.values = self.values[:(1 << len(self.qureg))]
        qubit.tangle = None
        qubit.measured = result
        qubit.prob = prob.value
        # print(qubit.prob, prob.value, result)

    def deal_reset_gate(self, gate):
        """ apply a reset gate on this tangle

        Note that after flush the reset gate, the qubit will be removed
        from the tangle.

        Args:
            gate(ResetGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
        """
        dll = systemCdll.quick_operator_cdll
        reset_operator_func = dll.reset_operator_func
        reset_operator_func.argtypes = [
            c_int,
            c_int,
            np.ctypeslib.ndpointer(dtype=np.complex, ndim=1, flags="C_CONTIGUOUS"),
        ]

        index = 0
        qubit = self.qureg.circuit.qubits[gate.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            index = index + 1
        if index == len(self.qureg):
            raise FrameworkException("the index is out of range")
        reset_operator_func(
            len(self.qureg),
            index,
            self.values
        )
        self.qureg.remove(qubit)
        self.values = self.values[:(1 << len(self.qureg))]
        qubit.tangle = None

    def deal_control_single_gate(self, gate):
        """ apply a controlled one qubit gate on this tangle

        Args:
            gate(BasicGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        qubit = self.qureg.circuit.qubits[gate.carg]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex = cindex + 1
        if cindex == len(self.qureg):
            raise FrameworkException("the index is out of range")

        tindex = 0
        qubit = self.qureg.circuit.qubits[gate.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            tindex = tindex + 1
        if tindex == len(self.qureg):
            raise FrameworkException("the index is out of range")
        # print("before:", np.round(self.values, decimals = 2))
        # print(cindex, tindex)
        control_single_operator_func(
            len(self.qureg),
            cindex,
            tindex,
            self.values,
            gate.matrix
        )
        # print("after:", np.round(self.values, decimals=2))

    def deal_ccx_gate(self, gate):
        """ apply a toffoli gate on this tangle

        Args:
            gate(BasicGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        qubit = self.qureg.circuit.qubits[gate.cargs[0]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex1 = cindex1 + 1
        if cindex1 == len(self.qureg):
            raise FrameworkException("the index is out of range")

        cindex2 = 0
        qubit = self.qureg.circuit.qubits[gate.cargs[1]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex2 = cindex2 + 1
        if cindex2 == len(self.qureg):
            raise FrameworkException("the index is out of range")

        tindex = 0
        qubit = self.qureg.circuit.qubits[gate.targ]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            tindex = tindex + 1
        if tindex == len(self.qureg):
            raise FrameworkException("the index is out of range")
        ccx_single_operator_func(
            len(self.qureg),
            cindex1,
            cindex2,
            tindex,
            self.values
        )

    def deal_swap_gate(self, gate):
        """ apply a swap gate on this tangle

        Args:
            gate(SwapGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
        """
        cindex = 0
        qubit = self.qureg.circuit.qubits[gate.targs[0]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            cindex = cindex + 1
        if cindex == len(self.qureg):
            raise FrameworkException("the index is out of range")

        tindex = 0
        qubit = self.qureg.circuit.qubits[gate.targs[1]]
        for test in self.qureg:
            if test.id == qubit.id:
                break
            tindex = tindex + 1
        if tindex == len(self.qureg):
            raise FrameworkException("the index is out of range")

        t = self.qureg[cindex]
        self.qureg[cindex] = self.qureg[tindex]
        self.qureg[tindex] = t

    def deal_custom_gate(self, gate):
        """ apply a custom gate on this tangle

        Args:
            gate(CustomGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        for idx in gate.targs:
            qubit = self.qureg.circuit.qubits[idx]
            temp_idx = 0
            for test in self.qureg:
                if test.id == qubit.id:
                    break
                temp_idx = temp_idx + 1
            if temp_idx == len(self.qureg):
                raise FrameworkException("the index is out of range")
            np.append(index, temp_idx)

        custom_operator_gate(
            len(self.qureg),
            self.values,
            index,
            gate.targets,
            gate.matrix
        )

    def deal_perm_gate(self, gate):
        """ apply a Perm gate on this tangle

        Args:
            gate(PermGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        targs = gate.targs
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
                raise FrameworkException("the index is out of range")
            index = np.append(index, temp_idx)
        perm_operator_gate(
            len(self.qureg),
            self.values,
            index,
            gate.targets,
            np.array(gate.pargs, dtype=np.int)
        )

    def deal_controlMulPerm_gate(self, gate):
        """ apply a controlMulPerm gate on this tangle

        Args:
            gate(controlMulPerm): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        targs = gate.targs
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
                raise FrameworkException("the index is out of range")
            index = np.append(index, temp_idx)
        control = gate.cargs[0]
        qubit = self.qureg.circuit.qubits[control]
        temp_idx = 0
        for test in self.qureg:
            if test.id == qubit.id:
                break
            temp_idx = temp_idx + 1
        if temp_idx == len(self.qureg):
            raise FrameworkException("the index is out of range")
        control = temp_idx
        perm_operator_gate(
            len(self.qureg),
            self.values,
            index,
            control,
            gate.targets,
            gate.pargs[0],
            gate.pargs[1]
        )

    def deal_shorInitial_gate(self, gate):
        """ apply a shorInitial gate on this tangle

        Args:
            gate(shorInitialGate): the gate to be applied.

        Exceptions:
            FrameworkException: the index is out of range
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
        targs = gate.targs
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
                raise FrameworkException("the index is out of range")
            index = np.append(index, temp_idx)
        perm_operator_gate(
            len(self.qureg),
            self.values,
            index,
            gate.targets,
            gate.pargs[0],
            gate.pargs[1],
            gate.pargs[2]
        )

    def force_assign_random(self):
        """ assign random values to the tangle

        """
        self.values = np.zeros(1 << len(self.qureg), dtype=np.complex)
        sqrnorm = 0
        for i in range(1 << len(self.qureg)):
            real = random.random()
            imag = random.random()
            self.values[i] = real + imag * 1j
            norm = abs(self.values[i])
            sqrnorm += norm * norm
        sqrnorm = sqrt(sqrnorm)
        for i in range(1 << len(self.qureg)):
            self.values[i] /= sqrnorm

    def force_assign_zeros(self):
        """ assign zero to the tangle

        """
        self.values = np.zeros(1 << len(self.qureg), dtype=np.complex)
        self.values[0] = 1 + 0j

    def force_copy(self, other):
        """ copy other tangle's values

        Args:
            other(Tangle): the item to be copied.

        """
        self.values = other.values.copy()

    def partial_prob(self, indexes):
        """ calculate the probabilities of the measure result of partial qureg in tangle

        Note that this function is a cheat function, which do not change the state of the qureg.

        Args:
            indexes(list<int>): the indexes of the partial qureg.

        Returns:
            list<float>: the probabilities of the measure result, the memory mode is LittleEndian.

        """
        back = [0.0] * (1 << len(indexes))
        save_list = []
        for id in indexes:
            for index in range(len(self.qureg)):
                if self.qureg[index].id == id:
                    save_list.append(index)
        lv = len(self.values)
        lq = len(self.qureg)
        for i in range(lv):
            pos = 0
            for j in range(len(save_list)):
                if (i & (1 << (lq - 1 - save_list[j]))) != 0:
                    pos += (1 << j)
            norm = abs(self.values[i])
            back[pos] += norm * norm
        return back
