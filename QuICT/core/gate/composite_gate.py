#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
from __future__ import annotations

from typing import Union, List
import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.utils import CircuitBased, CircuitMatrix, CGATE_LIST
from QuICT.tools.exception.core import CompositeGateAppendError, TypeError, GateQubitAssignedError


class CompositeGate(CircuitBased):
    """ Implement a group of gate """
    @property
    def qubits(self) -> list:
        return self._gates.qubits

    def __init__(
        self,
        name: str = None,
        gates: List[BasicGate, CompositeGate] = None,
        precision: str = "double"
    ):
        """
        Args:
            name (str, optional): the name of the composite gate. Defaults to None.
            gates (List[BasicGate, CompositeGate], optional): gates within this composite gate. Defaults to None.
        """
        super().__init__(name, precision=precision)
        if gates is not None:
            for gate in gates:
                if isinstance(gate, CompositeGate):
                    self.extend(gate)
                else:
                    self.append(gate)

    def clean(self):
        """ Remove all quantum gates in current CompositeGate. """
        self._gates.reset()
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

        self._qubit_indexes_validation(indexes)
        self._pointer = indexes
        return self

    def __and__(self, targets: Union[int, list]):
        """ assign indexes for the composite gates

        Args:
            targets ([int/list[int]]): qubit describe
        """
        if isinstance(targets, int):
            targets = [targets]

        self._qubit_indexes_validation(targets)
        self._gates.reassign(targets)

        if CGATE_LIST:
            CGATE_LIST[-1].extend(self)

        return self

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
                2) CompositeGate
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

        return gate.copy() & qidxes

    def extend(self, gates: CompositeGate):
        """ Add a CompositeGate to current CompositeGate.

        Args:
            gates (CompositeGate): The given CompositeGate
        """
        if self._pointer is not None:
            gate_args = gates.width()
            assert gate_args <= len(self._pointer), GateQubitAssignedError(
                f"{gates.name} need at least {gate_args} indexes, but given {len(self._pointer)}"
            )
            if gate_args == len(self._pointer):
                gate_qidxes = self._pointer[:]
            else:
                gate_qidxes = [self._pointer[qidx] for qidx in gates.qubits]
        else:
            self._qubit_indexes_validation(gates.qubits)
            gate_qidxes = gates.qubits

        self._gates.extend(gates, gate_qidxes)
        self._pointer = None

    def append(self, gate: BasicGate):
        """ Add a quantum gate to current CompositeGate.

        Args:
            gate (BasicGate): The quantum gate need to append
        """
        if not isinstance(gate, BasicGate):
            raise TypeError("CompositeGate.append", "BasicGate/CheckPoint", type(gate))

        if self._pointer is not None:
            gate_args = gate.controls + gate.targets
            assert len(self._pointer) == gate_args, \
                GateQubitAssignedError(f"{gate.type} need {gate_args} indexes, but given {len(self._pointer)}")

            qubit_index = self._pointer[:]
        else:
            qubit_index = gate.cargs + gate.targs
            if not qubit_index:
                raise GateQubitAssignedError(f"{gate.type} need qubit indexes to add into Composite Gate.")

            self._qubit_indexes_validation(qubit_index)

        self._gates.append(gate, qubit_index)
        self._pointer = None

    ####################################################################
    ############            CompositeGate Utils             ############
    ####################################################################
    def inverse(self) -> CompositeGate:
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        _gates = CompositeGate()
        for gate in self._gates.gates[::-1]:
            if isinstance(gate, CompositeGate):
                _gates.extend(gate.inverse())
            else:
                _gates._gates.append(gate.gate.inverse(), gate.indexes)

        return _gates

    def copy(self) -> CompositeGate:
        """ Copy current CompositeGate. """
        _gates = CompositeGate()
        _gates.name = self.name
        _gates._gates = self._gates.copy()

        return _gates

    def matrix(self, device: str = "CPU", local: bool = False) -> np.ndarray:
        """ matrix of these gates

        Args:
            device (str, optional): The device type for generate circuit's matrix, one of [CPU, GPU]. Defaults to "CPU".
            local: whether consider only about the occupied qubits or not

        Returns:
            np.ndarray: the matrix of the gates
        """
        assert device in ["CPU", "GPU"]
        circuit_matrix = CircuitMatrix(device, self._precision)
        if not local:
            matrix_width = max(self.qubits) + 1
            assigned_gates = self.flatten_gates()
        else:
            matrix_width = self.width()
            based_qubits, assigned_gates = self.qubits, []
            for gate, qidx in self._gates.tree_search():
                new_qidx = [based_qubits.index(q) for q in qidx]
                assigned_gates.append(gate.copy() & new_qidx)

        return circuit_matrix.get_unitary_matrix(assigned_gates, matrix_width)
