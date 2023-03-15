#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
from __future__ import annotations

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
        gates (list<Tuple[Gate/Operator, qubits, size]>): gates within this composite gate
    """
    @property
    def qubits(self) -> list:
        return self._qubits

    @property
    def checkpoint(self):
        return self._check_point

    def __init__(self, name: str = None, gates: list = None):
        if name is None:
            name = "composite_gate_" + unique_id_generator()

        super().__init__(name)
        self._check_point = None        # required checkpoint
        self._qubits = []

        if gates is not None:
            self.extend(gates)

    def clean(self):
        self._gates = []
        self._qubits = []
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
    def __call__(self, indexes: Union[list, int]):
        if isinstance(indexes, int):
            indexes = [indexes]

        self._pointer = indexes
        return self

    def __and__(self, targets: Union[int, list]):
        """ assign indexes for the composite gates

        Args:
            targets ([int/list[int]]): qubit describe
        """
        if isinstance(targets, int):
            targets = [targets]

        if len(targets) != self.width():
            raise ValueError("CompositeGate.&", f"not equal {self.width}", len(targets))

        self._mapping(targets)

    def _mapping(self, targets: list):
        """ remapping the gates' target qubits

        Args:
            targets(list): the related qubits
        """
        qidx_mapping = {}
        for i, q in enumerate(self._qubits):
            qidx_mapping[q] = targets[i]

        new_gates = []
        for gate, qidxes, _ in self._gates:
            new_q = [qidx_mapping[qidx] for qidx in qidxes]
            new_gates.append((gate, new_q))

        self._qubits = targets

    def _update_qubit_limit(self, indexes: list):
        for idx in indexes:
            assert idx >= 0 and isinstance(idx, int)
            if idx not in self._qubits:
                self._qubits.append(idx)

    ####################################################################
    ############            CompositeGate Build             ############
    ####################################################################
    def __or__(self, targets):
        """ deal the operator '|'

        Use the syntax "CompositeGate | circuit", "CompositeGate | CompositeGate"
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

    def extend(self, gates: CompositeGate, reverse: bool = False):
        if self._pointer is not None:
            gate_args = gates.width()
            assert gate_args == len(self._pointer), GateQubitAssignedError(
                f"{gates.name} need {gate_args} indexes, but given {len(self._pointer)}"
            )

            gates & self._pointer

        if not reverse:
            self._gates.append((gates, gates.qubits, gates.size()))
        else:
            self._gates.insert(0, (gates, gates.qubits, gates.size()))

        print(gates.qubits)
        self._update_qubit_limit(gates.qubits)
        self._pointer = None

    def append(self, gate):
        if isinstance(gate, BasicGate):
            self._append_gate(gate)
        elif isinstance(gate, CheckPointChild):
            self._check_point = gate
        else:
            raise TypeError("CompositeGate.append", "BasicGate/CheckPoint", type(gate))

        self._pointer = None

    def insert(self, gate, insert_idx: int):
        """ Insert a Quantum Gate into current CompositeGate, only support BasicGate. """
        assert isinstance(gate, BasicGate), TypeError("CompositeGate.insert", "BasicGate", type(gate))
        gate_args = gate.cargs + gate.targs
        if len(gate_args) == 0:
            raise GateQubitAssignedError(f"{gate.type} need qubit indexes to insert into Composite Gate.")

        self._update_qubit_limit(gate_args)
        self._gates.insert(insert_idx, (gate, gate_args, 1))

    def _append_gate(self, gate: BasicGate):
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
        self._gates.append((gate, qubit_index, 1))

    ####################################################################
    ############            CompositeGate Utils             ############
    ####################################################################
    def inverse(self):
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        self._gates = [(gate.inverse(), indexes, size) for gate, indexes, size in self._gates[::-1]]

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
