#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 8:44
# @Author  : Han Yu
# @File    : _qubit.py

import random

from QuICT.core.exception import *
from QuICT.core.utils import unique_id_generator


class Qubit(object):
    """ Implement a quantum bit

    Qubit is the basic unit of quantum circuits, it will appear with some certain circuit.

    Attributes:
        id(int): the unique identity code of a qubit, which is generated globally.
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
        return self._id

    @property
    def measured(self) -> int:
        return self._measured

    @measured.setter
    def measured(self, measured):
        self._measured = measured

    @property
    def prob(self) -> float:
        return self._prob

    @prob.setter
    def prob(self, prob):
        self._prob = prob

    def __init__(self, prob: float = None):
        """ initial a qubit with a circuit

        Args:
            circuit(Circuit): the circuit the qubit attaches to
        """
        self._id = unique_id_generator()
        self._measured = -1
        self._prob = random.random() if prob is None else prob

    def __str__(self):
        """ string describe of the qubit

        Returns:
            str: a simple describe
        """
        return f"qubit id: {self.id}"

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
        else:
            return bool(self.measured)


class Qureg(list):
    """ Implement a quantum register

    Qureg is a list of Qubits, which is a subClass of list.

    Attributes:
        qubits([Qubits]): the list of qubits.
    """
    def __init__(self, qubits):
        """ initial a qureg with qubit(s)

        Args:
            qubits: the qubits which make up the qureg, it can have below form,
                1) int
                2) [qubits/quregs]
            circuit_name (str): the circuit's name which this qureg belong to.
        """
        super().__init__()

        if isinstance(qubits, int):
            for _ in range(qubits):
                self.append(Qubit())
        elif isinstance(qubits, list):
            for qubit in qubits:
                if isinstance(qubit, Qubit):
                    self.append(qubit)
                elif isinstance(qubit, Qureg):
                    for qbit in qubit:
                        self.append(qbit)
                else:
                    raise TypeException("list<Qubits/Qureg>", qubits)
        else:
            raise TypeException("list<Qubits/Qureg> or int", qubits)

    def __call__(self, indexes: object):
        """ get a smaller qureg from this qureg

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) list<int>
        Returns:
            Qubit[s]: the qureg correspond to the indexes
        Exceptions:
            IndexDuplicateException: the range of indexes is error.
            TypeException: the type of indexes is error.
        """
        if isinstance(indexes, int):        # int
            if indexes < 0 or indexes >= len(self):
                raise IndexLimitException(len(self), indexes)
        elif isinstance(indexes, list):     # list
            if len(indexes) != len(set(indexes)):
                raise IndexDuplicateException(indexes)

            for element in indexes:
                if not isinstance(element, int):
                    raise TypeException("int", element)
                if element < 0 or element >= len(self):
                    raise IndexLimitException(len(self), element)
        else:
            raise TypeException("int or list", indexes)

        return self[indexes]

    def __int__(self):
        """ the value of the register

        Return the value of the register if all qubits have been measured.
        Note that the compute mode is BigEndian.

        Returns:
            int: the value of the register

        Raises:
            Exception: some qubit has not be measured
        """
        value = 0
        for qubit in self:
            if qubit.measured == -1:
                raise Exception(f"The qubit {qubit.id} has not be measured")

            value <<= 1
            if qubit.measured == 1:
                value += 1

        return value

    def __str__(self):
        """ a simple describe

        Returns:
            str: the value of the qureg
        """
        bit_idx = "{0:0b}".format(self.__int__())
        bit_idx.zfill(len(self))

        return bit_idx

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

    def __eq__(self, other):
        current_qubit_ids = [qubit.id for qubit in self]
        for qubit in other:
            if qubit.id not in current_qubit_ids:
                return False

        return True

    def diff(self, other):
        others_qubit_ids = [qubit.id for qubit in other]
        diff_qubit = []

        for qubit in self:
            if qubit.id not in others_qubit_ids:
                diff_qubit.append(qubit)

        return diff_qubit
