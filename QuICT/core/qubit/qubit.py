#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 8:44
# @Author  : Han Yu
# @File    : _qubit.py

from math import sqrt
import random
import weakref

import numpy as np

from ..exception import *

# global qubit id count
qubit_id = 0

# global qState id count
QState_id = 0

class Qubit(object):
    """ Implement a quantum bit

    Qubit is the basic unit of quantum circuits, it will appear with some certain circuit.

    Attributes:
        id(int): the unique identity code of a qubit, which is generated globally.
        qState(QState):
            a special qureg in which all qubits may enqState with each other.
            class QState will be defined and introduced below.
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

    @property
    def qState(self):
        if self.__qState is None:
            self.__qState = QState(self)
        return self.__qState

    @qState.setter
    def qState(self, qState):
        self.__qState = qState

    @property
    def circuit(self):
        return self.__circuit()

    @circuit.setter
    def circuit(self, circuit):
        self.__circuit = weakref.ref(circuit)

    @property
    def measured(self) -> int:
        return self.__measured

    @measured.setter
    def measured(self, measured):
        self.__measured = measured

    @property
    def prob(self) -> float:
        return self.__prob

    @prob.setter
    def prob(self, prob):
        self.__prob = prob

    def __init__(self, circuit):
        """ initial a qubit with a circuit

        Args:
            circuit(Circuit): the circuit the qubit attaches to
        """
        global qubit_id
        self.__id = qubit_id
        qubit_id = qubit_id + 1
        self.circuit = circuit
        self.__qState = None
        self.__measured = -1
        self.__prob = 0.0

    def __str__(self):
        """ string describe of the qubit

        Returns:
            str: a simple describe
        """
        return f"circuit:{self.circuit} id:{self.id}"

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

    def __getitem__(self, item):
        """ to fit the the operator qubit[0], return itself

        Args:
            item(int): must be 0
        Return:
            Qubit: itself
        """
        if isinstance(item, int) and item == 0:
            return self
        raise Exception("the item passes to Qubit should be 0.")

    def __call__(self, indexes: object):
        """ get a smaller qureg from this qureg

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) [0]
                3) (0)
        Returns:
            Qureg: the qureg correspond to self
        Exceptions:
            IndexDuplicateException: the range of indexes is error.
            TypeException: the type of indexes is error.
        """

        # int
        if isinstance(indexes, int):
            if indexes == 0:
                return Qureg(self)

        # tuple
        if isinstance(indexes, tuple):
            if len(indexes) == 1 and indexes[0] == 0:
                return Qureg(self)

        # list
        if isinstance(indexes, list):
            if len(indexes) == 1 and indexes[0] == 0:
                return Qureg(self)

        raise Exception("the item passes to Qubit should be 0.")

    def has_qState(self):
        """ whether the qubit is belonged to a qState.

        Returns:
            bool: True if qubit is belonged to a qState
        """
        if self.__qState is None:
            return False
        else:
            return True

    def qState_clear(self):
        """ delete the qState of the qubit

        """
        if self.__qState is not None:
            del self.__qState
        self.__qState = None
        self.__measured = -1

class Qureg(list):
    """ Implement a quantum register

    Qureg is a list of Qubit, which is a subClass of list.

    Attributes:
        circuit(Circuit): the circuit this qureg belonged to.
    """

    @property
    def circuit(self):
        if len(self) == 0:
            raise Exception("illegal operation，the qureg is empty.")
        return self[0].circuit

    @circuit.setter
    def circuit(self, circuit):
        if len(self) == 0:
            raise Exception("illegal operation，the qureg is empty.")
        for qubit in self:
            qubit.circuit = circuit

    def __init__(self, qubits = None):
        """ initial a qureg with qubit(s)

        Args:
            qubits: the qubits which make up the qureg, it can have below form,
                1) Circuit
                2) Qureg
                3) Qubit
                4) tuple/list<Qubit/Qureg>
        """
        from QuICT import Circuit
        super().__init__()
        if qubits is None:
            return
        if isinstance(qubits, Qubit):
            self.append(qubits)
        elif isinstance(qubits, tuple) or isinstance(qubits, list):
            for qubit in qubits:
                if isinstance(qubit, Qubit):
                    self.append(qubit)
                elif isinstance(qubit, Qureg):
                    for qbit in qubit:
                        self.append(qbit)
                else:
                    raise TypeException("Qubit/tuple<Qubit> or Qureg or Circuit", qubits)

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

    def __add__(self, other):
        """ to fit the add operator, overloaded this function.

        get a smaller qureg/qubit from this qureg

        Args:
            other(Qureg): qureg to be added.
        Return:
            Qureg: the result or slice
        """
        if not isinstance(other, Qureg):
            raise Exception("type error!")
        qureg_list = super().__add__(other)
        return Qureg(qureg_list)

    def force_assign_random(self):
        """ assign random values for qureg which has initial values

        after calling this function, all qubits will be in a QState.

        """
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_qState():
                raise Exception(f"qubit {qubit.id} has initial value")
        qState = QState(self[0])
        for i in range(1, len(self)):
            qState.qureg.append(self[i])
        qState.force_assign_random()
        for qubit in self:
            qubit.qState = qState

    def force_assign_zeros(self):
        """ assign zero for qureg has initial values

        after calling this function, all qubits will be in a QState.

        """
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_qState():
                raise Exception(f"qubit {qubit.id} has initial value")
        qState = QState(self[0])
        for i in range(1, len(self)):
            qState.qureg.append(self[i])
        qState.force_assign_zeros()
        for qubit in self:
            qubit.qState = qState

    def force_copy(self, copy_item, indexes):
        """ copy values from other QState for this qureg which has initial values

        after calling this function, all qubits will be in a QState.

        Args:
            copy_item(QState): the QState need to be copied.
            indexes(list<int>): the indexes of goal qubits

        """
        if len(self) == 0:
            return
        for qubit in self:
            if qubit.has_qState():
                raise Exception(f"qubit {qubit.id} has initial value")
        qState = QState(self[indexes[0]])
        self[indexes[0]].qState = qState
        for i in range(1, len(indexes)):
            qState.qureg.append(self[indexes[i]])
            self[indexes[i]].qState = qState
        qState.force_copy(copy_item)

class QState(object):
    """ Implement a QState

    a basic computation unit of the amplitude simulation, in which
    all qubits are seemed enqState with each other.

    Attributes:
        id(int): the unique identity code of a qState, which is generated globally.
        qureg(Qureg): the qubit register of the qState
        values(np.array): the inner amplitude of this qState

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
        """ initial a qState with one qubit

        Args:
            qubit: the qubit form the qState.

        """
        self.__qureg = Qureg(qubit)
        if qubit.measured == -1 or qubit.measured == 0:
            self.__values = np.array([1, 0], dtype=np.complex128)
        else:
            self.__values = np.array([0, 1], dtype=np.complex128)
        global QState_id
        self.__id = QState_id
        QState_id = QState_id + 1

    def __del__(self):
        """ release the memory

        """
        self.values = None
        self.qureg = None

    def print_information(self):
        """ print the infomation of the qState

        """
        print(self.values, self.id, len(self.qureg))

    def index_for_qubit(self, qubit) -> int:
        """ find the index of qubit in this qState's qureg

        Args:
            qubit(Qubit): the qubit need to be indexed.

        Returns:
            int: the index of the qubit.

        Raises:
            Exception: the qubit is not in the qState
        """
        if not isinstance(qubit, Qubit):
            raise TypeException("Qubit", qubit)
        for i in range(len(self.qureg)):
            if self.qureg[i].id == qubit.id:
                return i
        raise Exception("the qubit is not in the qState")

    # cheat methods
    def force_assign_random(self):
        """ assign random values to the qState

        """
        self.values = np.zeros(1 << len(self.qureg), dtype=np.complex128)
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
        """ assign zero to the qState

        """
        self.values = np.zeros(1 << len(self.qureg), dtype=np.complex128)
        self.values[0] = 1 + 0j

    def force_copy(self, other):
        """ copy other qState's values

        Args:
            other(QState): the item to be copied.

        """
        self.values = other.values.copy()

    def partial_prob(self, indexes):
        """ calculate the probabilities of the measure result of partial qureg in qState

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
