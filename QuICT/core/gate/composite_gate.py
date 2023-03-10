#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
from typing import Union
import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.operator import CheckPointChild, Operator
from QuICT.core.utils import (
    CircuitBased,
    CGATE_LIST,
    unique_id_generator
)
from QuICT.tools.exception.core import ValueError, CompositeGateAppendError, TypeError, GateQubitAssignedError


class CompositeGate(CircuitBased):
    """ Implement a group of gate

    Attributes:
        name (str): the name of the composite gate
        gates (list<BasicGate>): gates within this composite gate
    """
    @property
    def checkpoint(self):
        return self._check_point

    def __init__(self, name: str = None, gates: list = None):
        if name is None:
            name = "composite_gate_" + unique_id_generator()

        super().__init__(name)
        self._check_point = None        # required checkpoint
        self._qubit_indexes = []

        if gates is not None:
            self.extend(gates)

    def clean(self):
        self._gates = []
        self._qubit_indexes = []
        self._pointer = None

    ####################################################################
    ############          CompositeGate Context             ############
    ####################################################################
    def __enter__(self):
        global CGATE_LIST
        CGATE_LIST.append(self)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # print(f"{exc_type}: {exc_value}")
        if exc_type is not None:
            raise Exception(exc_value)

        global CGATE_LIST
        CGATE_LIST.remove(self)

        return True

    ####################################################################
    ############        CompositeGate Qureg Mapping         ############
    ####################################################################
    def width(self):
        """ the number of qubits applied by gates

        Returns:
            int: the number of qubits applied by gates
        """
        return len(self._qubit_indexes)

    def __call__(self, indexes: Union[list, int]):
        if isinstance(indexes, int):
            indexes = [indexes]

        self._update_qubit_limit(indexes)
        self._pointer = indexes
        return self

    def __and__(self, targets: Union[int, list]):
        """ assign qubits or indexes for given gates

        Args:
            targets ([int/list[int]]): qubit describe
        """
        if isinstance(targets, int):
            targets = [targets]

        if len(targets) != self.width():
            raise ValueError("CompositeGate.&", f"not equal {self.width}", len(targets))

        self._mapping(targets)

    def _mapping(self, targets: list):
        """ remapping the gates' affectArgs

        Args:
            targets(list): the related qubits
        """
        qidx_mapping = {}
        for i, q in enumerate(self._qubit_indexes):
            qidx_mapping[q] = targets[i]

        new_gates = []
        for gate, qidxes in self._gates:
            new_q = [qidx_mapping[qidx] for qidx in qidxes]
            new_gates.append((gate, new_q))

        self._qubit_indexes = targets

    def _update_qubit_limit(self, indexes: list):
        for idx in indexes:
            assert idx >= 0 and isinstance(idx, int)
            if idx not in self._qubit_indexes:
                self._qubit_indexes.append(idx)

    ####################################################################
    ############            CompositeGate Build             ############
    ####################################################################
    def __or__(self, targets):
        """ deal the operator '|'

        Use the syntax "gateSet | circuit", "gateSet | gateSet"
        to add the gate of gateSet into the circuit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) CompositeGate
        Raise:
            TypeError: the type of other is wrong
        """
        try:
            targets.extend(self)
        except Exception as e:
            raise CompositeGateAppendError(f"Failure to append current CompositeGate, due to {e}.")

    def __xor__(self, targets):
        """deal the operator '^'

        Use the syntax "gateSet ^ circuit", "gateSet ^ gateSet"
        to add the gate of gateSet's inverse into the circuit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
        Raise:
            TypeError: the type of other is wrong
        """
        try:
            targets.extend(self.inverse())
        except Exception as e:
            raise CompositeGateAppendError(f"Failure to append the inverse of current CompositeGate, due to {e}.")

    def __getitem__(self, item):
        """ get gates from this composite gate

        Args:
            item(int/slice): slice passed in.

        Return:
            [BasicGates]: the gates
        """
        gate, qidxes = self._gates[item]

        return gate & qidxes

    def extend(self, gates: list):
        for gate in gates:
            self.append(gate, is_extend=True)

        self._pointer = None

    def append(self, gate, is_extend: bool = False):
        if isinstance(gate, BasicGate):
            self._append_gate(gate)
            if not is_extend:
                self._pointer = None
        elif isinstance(gate, CheckPointChild):
            self._check_point = gate
        else:
            assert isinstance(gate, Operator), TypeError("CompositeGate.append", "BasicGate/Operator", type(gate))
            if insert_idx == -1:
                self._gates.append(gate)
            else:
                self._gates.insert(insert_idx, gate)

    def insert(self, gate, insert_idx: int):
        """ Only support BasicGate """
        assert isinstance(gate, BasicGate)
        gate_args = gate.cargs + gate.targs
        if len(gate_args) == 0:
            raise ValueError("Error, replace with CompositeGateInsertError.")

        self._update_qubit_limit(gate_args)
        self._gates.insert(insert_idx, (gate, gate_args))

    def left_extend(self, gates: list):
        for idx, gate in enumerate(gates):
            self.insert(gate, insert_idx=idx)

    def _append_gate(self, gate):
        if self._pointer is not None:
            gate_args = gate.controls + gate.targets
            if len(self._pointer) < gate_args:
                raise GateQubitAssignedError(f"{gate.type} need {gate_args} indexes, but given {len(self._pointer)}")

            if len(self._pointer) > gate_args:
                qubit_index = [self._pointer[qarg] for qarg in gate.cargs + gate.targs]
            else:
                qubit_index = self._pointer[:]
        else:
            qubit_index = gate.cargs + gate.targs
            if not qubit_index:
                raise GateQubitAssignedError(f"{gate.type} need qubit indexes to add into Composite Gate.")

            self._update_qubit_limit(qubit_index)

        self._gates.append((gate, qubit_index))

    ####################################################################
    ############            CompositeGate Utils             ############
    ####################################################################
    def inverse(self):
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        self._gates = [(gate.inverse(), indexes) for gate, indexes in self._gates[::-1]]

        return self

    # TODO: return matrix only with used qubits
    def matrix(self, device: str = "CPU", local: bool = False) -> np.ndarray:
        """ matrix of these gates

        Args:
            device (str, optional): The device type for generate circuit's matrix, one of [CPU, GPU]. Defaults to "CPU".
            local: whether consider only about 

        Returns:
            np.ndarray: the matrix of the gates
        """
        if local and isinstance(self._min_qubit, int):
            min_value = self._min_qubit
        else:
            min_value = 0

        return super().matrix(device, min_value)
