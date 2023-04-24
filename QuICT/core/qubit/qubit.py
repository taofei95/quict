#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 8:44
# @Author  : Han Yu, Li Kaiqi
# @File    : qubit.py
from __future__ import annotations
from typing import Union, List
from collections import defaultdict

from QuICT.core.utils import unique_id_generator
from QuICT.tools.exception.core import TypeError, ValueError, IndexExceedError, QubitMeasureError


class Qubit(object):
    """ Implement a Quantum bit

    Qubit is the basic unit of Quantum Compute.
    """

    @property
    def id(self):
        return self._id

    @property
    def measured(self) -> int:
        return self._measured

    @measured.setter
    def measured(self, measured: int):
        assert measured == 0 or measured == 1, ValueError("Qubit.measured", "one of [0, 1]", measured)
        self._measured = measured
        self._historical_measured.append(measured)

    @property
    def historical_measured(self) -> list:
        return self._historical_measured

    @property
    def fidelity(self) -> float:
        return self._fidelity

    @fidelity.setter
    def fidelity(self, fidelity: float):
        assert isinstance(fidelity, (float, int)), TypeError("Qubit.fidelity", "float, int", type(fidelity))
        assert fidelity >= 0 and fidelity <= 1, ValueError("Qubit.fidelity", "within [0, 1]", {fidelity})

        self._fidelity = fidelity

    @property
    def T1(self) -> float:
        return self._t1

    @T1.setter
    def T1(self, t1: float):
        assert isinstance(t1, (float, int)) and t1 >= 0, ValueError("Qubit.T1", "greater than 0", t1)
        self._t1 = t1

    @property
    def T2(self) -> float:
        return self._t2

    @T2.setter
    def T2(self, t2: float):
        assert isinstance(t2, (float, int)) and t2 >= 0, ValueError("Qubit.T2", "greater than 0", t2)
        self._t2 = t2

    def __init__(self, fidelity: float = 1.0, T1: float = 0.0, T2: float = 0.0):
        """ initial a qubit

        Args:
            fidelity (float): The qubit's fidelity, where the fidelity of a quantum qubit is the overlap between
                the ideal theoretical operation and the actual experimental operation.
            T1 (float, μs): The longitudinal coherence time, which refers to the time it takes for the qubit to decay
                back to its ground state from an excited state. Default to None.
            T2 (float, μs): the transverse coherence time, which refers to the time it takes for the qubit to lose its
                coherence when subjected to unwanted phase or amplitude fluctuations. Default to None.
        """
        self._id = unique_id_generator()
        self.fidelity = fidelity
        self.T1 = T1
        self.T2 = T2

        self._measured = None
        self._historical_measured = []

    def __str__(self):
        """ string describe of the qubit

        Returns:
            str: a simple describe
        """
        return f"qubit id: {self.id}; fidelity: {self.fidelity}; Coherence time: T1: {self._t1}; T2: {self._t2}."

    def __int__(self):
        """ int value of the qubit(measure result)

        Returns:
            int: measure result

        Raises:
            The qubit has not be measured.
        """
        if self._measured is None:
            raise QubitMeasureError(f"The qubit {self.id} has not be measured")

        return self.measured

    def __bool__(self):
        """ int value of the qubit(measure result)

        Returns:
            bool: measure result

        Raises:
            The qubit has not be measured.
        """
        return bool(int(self))

    def reset(self):
        """ Reset self qubit status. """
        self._measured = None
        self._historical_measured = []


class Qureg(list):
    """ Implement a Quantum Register

    Qureg is a list of Qubits, which is a subClass of list.
    """
    @property
    def coupling_strength(self) -> list:
        return self._coupling_strength

    def __init__(self, qubits: Union[int, Qubit, Qureg] = None, coupling_strength: list = None):
        """ initial a qureg with qubit(s)

        Args:
            qubits: the qubits which make up the qureg, it can have below form,
                1) int
                2) qubit
                3) [qubits/quregs]
            coupling_strength List[Tuple(idx, idx, float)]: The strength of the interaction between two qubits in a
                quantum computing system. It should follow by the physical topology.
        """
        super().__init__()
        if qubits is None:
            return

        if isinstance(qubits, int):
            for _ in range(qubits):
                self.append(Qubit())
        elif isinstance(qubits, Qubit):
            self.append(qubits)
        elif isinstance(qubits, list):
            for qubit in qubits:
                if isinstance(qubit, Qubit):
                    self.append(qubit)
                elif isinstance(qubit, Qureg):
                    self.extend(qubit)
                else:
                    raise TypeError("Qureg.qubits", "int/Qubit/list<Qubit/Qureg>", type(qubit))
        else:
            raise TypeError("Qureg.qubits", "int/Qubit/list<Qubit/Qureg>", type(qubit))

        self._coupling_strength = defaultdict(dict)
        if coupling_strength is not None:
            self.set_coupling_strength(coupling_strength)

    def __call__(self, indexes: object):
        """ get a smaller qureg from this qureg

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) list<int>
        Returns:
            Qubit[s]: the qureg correspond to the indexes
        """
        if isinstance(indexes, int):
            return Qureg(self[indexes])

        return self[indexes]

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

        qureg = Qureg()
        if isinstance(item, slice):
            qureg_list = super().__getitem__(item)
            for qubit in qureg_list:
                qureg.append(qubit)
        elif isinstance(item, list) or isinstance(item, tuple):
            for idx in item:
                assert isinstance(idx, int), TypeError("Qureg.getitem.item.value", "int", type(idx))
                if idx < 0 or idx > len(self):
                    raise IndexExceedError("Qureg.getitem", [0, len(self)], idx)

                qureg.append(self[idx])
        else:
            raise TypeError("Qureg.getitem", "int/list[int]/slice", type(item))

        return qureg

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
            value <<= 1
            value += int(qubit)

        return value

    def __str__(self):
        """ the string of the value of the register

        Returns:
            str: the value of the qureg
        """
        bit_idx = "{0:0b}".format(self.__int__())
        bit_idx.zfill(len(self))

        return bit_idx

    def __add__(self, other):
        """ get a combined qureg with this qureg and other qureg

        Args:
            other(Qureg): qureg to be added.

        Return:
            Qureg: the result or slice
        """
        return Qureg([self, other])

    def __iadd__(self, other):
        """ get a combined qureg with this qureg and other qureg

        Args:
            other(Qureg): qureg to be added.

        Return:
            Qureg: the result or slice
        """
        if isinstance(other, Qubit):
            self.append(other)
        elif isinstance(other, Qureg):
            self.extend(other)
        else:
            raise TypeError("Qureg.iadd", "Qureg/Qubit", type(other))

        return self

    def __eq__(self, other):
        """
        check two qureg is same or not. Iff all qubits in two qureg are same will
        return True; otherwise, return False.

        Args:
            other(Qureg): qureg to be checked.
        """
        assert isinstance(other, Qureg), TypeError("Qureg.eq", "Qureg", type(other))
        if not len(other) == len(self):
            return False

        current_qubit_ids = [qubit.id for qubit in self]
        for qubit in other:
            if qubit.id not in current_qubit_ids:
                return False

        return True

    def index(self, qubit: Union[List[Qubit], Qubit]) -> Union[int, list]:
        """ Return the index of given qubits.

        Args:
            qubit (Union[List[Qubit], Qubit]): The given qubits

        Returns:
            Union[int, list]: The index of given qubits in current qureg.
        """
        if isinstance(qubit, Qubit):
            return super().index(qubit)

        if isinstance(qubit, list):
            idxes = []
            for q in qubit:
                idxes.append(super().index(q))

            return idxes

        raise ValueError("Qureg.index.qubit", "within current Qureg", "qubit is not")

    def set_fidelity(self, fidelity: list):
        """ Set the fidelity for each qubits

        Args:
            fidelity (list): The list of fidelity for each qubits, should equal to len(qureg).
        """
        assert isinstance(fidelity, list), \
            TypeError("Qureg.fidelity", "List", f"{type(fidelity)}")
        assert len(fidelity) == len(self), \
            ValueError("Qureg.fidelity", f"the length should equal {len(self)}", f"{len(fidelity)}")

        for idx, qubit in enumerate(self):
            qubit.fidelity = fidelity[idx]

    def set_t1_time(self, t1_time: list):
        """ Set the T1 coherence time for each qubit

        Args:
            t1_time (list): The T1 time for each qubit
        """
        assert isinstance(t1_time, list), \
            TypeError("Qureg.t1_time", "List", f"{type(t1_time)}")
        assert len(t1_time) == len(self), \
            ValueError("Qureg.t1_time", f"the length should equal {len(self)}", f"{len(t1_time)}")

        for idx, qubit in enumerate(self):
            qubit.T1 = t1_time[idx]

    def set_t2_time(self, t2_time: list):
        """ Set the T2 coherence time for each qubit

        Args:
            t2_time (list): The T2 time for each qubit
        """
        assert isinstance(t2_time, list), \
            TypeError("Qureg.t2_time", "List", f"{type(t2_time)}")
        assert len(t2_time) == len(self), \
            ValueError("Qureg.t2_time", f"the length should equal {len(self)}", f"{len(t2_time)}")

        for idx, qubit in enumerate(self):
            qubit.T2 = t2_time[idx]

    def set_coupling_strength(self, coupling_strength: list):
        """ Set the coupling strength between qubits

        Args:
            coupling_strength (list): The coupling strength, should be a 2D array with shape(len(qureg) * len(qureg))
        """
        for start, end, val in coupling_strength:
            assert start != end and start >= 0 and end >= 0 and start < len(self) and end < len(self)
            assert val >= 0 and val <= 1
            self._coupling_strength[start][end] = val
            self._coupling_strength[end][start] = val

    def reset_qubits(self):
        """ Reset all qubits' status. """
        for qubit in self:
            qubit.reset()
