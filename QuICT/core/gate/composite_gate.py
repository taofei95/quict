#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
from __future__ import annotations

from typing import Union, List
import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.utils import CircuitBased, CircuitMatrix, CGATE_LIST, unique_id_generator
from QuICT.tools.exception.core import CompositeGateAppendError, TypeError, GateQubitAssignedError


class CompositeGate(CircuitBased):
    """ Implement a group of gate """
    @property
    def qubits(self) -> list:
        return self._gates.qubits

    def __init__(self, name: str = None, gates: List[BasicGate, CompositeGate] = None):
        """
        Args:
            name (str, optional): the name of the composite gate. Defaults to None.
            gates (List[BasicGate, CompositeGate], optional): gates within this composite gate. Defaults to None.
        """
        if name is None:
            name = "composite_gate_" + unique_id_generator()

        super().__init__(name)
        if gates is not None:
            for gate in gates:
                if isinstance(gate, CompositeGate):
                    self.extend(gate)
                else:
                    self.append(gate)

    def clean(self):
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
    def _qubits_validation(self, qubit_indexes: list):
        for qidx in qubit_indexes:
            assert qidx >= 0 and isinstance(qidx, (int, np.int32, np.int64)), \
                "The gate's qubit indexes should be integer or greater than zero."

    def __call__(self, indexes: Union[list, int]):
        if isinstance(indexes, int):
            indexes = [indexes]

        self._qubits_validation(indexes)
        self._pointer = indexes
        return self

    def __and__(self, targets: Union[int, list]):
        """ assign indexes for the composite gates

        Args:
            targets ([int/list[int]]): qubit describe
        """
        if isinstance(targets, int):
            targets = [targets]

        self._qubits_validation(targets)
        print(self.qubits)
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

        return gate & qidxes

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

        self._gates.append(gate, qubit_index)
        self._pointer = None

    def insert(self, gate: Union[BasicGate, CompositeGate], insert_idx: int):
        """ Insert a Quantum Gate into current CompositeGate.

        Args:
            gate (Union[BasicGate, CompositeGate]): The quantum gate want to insert
            insert_idx (int): The index of insert position
        """
        assert isinstance(gate, (BasicGate, CompositeGate)), \
            TypeError("CompositeGate.insert", "BasicGate/CompositeGate", type(gate))

        if isinstance(gate, BasicGate):
            gate_args = gate.cargs + gate.targs
            gate_size = 1
        else:
            gate_args = gate.qubits
            gate_size = gate.size()

        if len(gate_args) == 0:
            raise GateQubitAssignedError(f"{gate.type} need qubit indexes to insert into Composite Gate.")

        self._gates.insert(insert_idx, (gate, gate_args, gate_size))

    def pop(self, index: int = -1):
        """ Pop the BasicGate/Operator/CompositeGate from current Quantum Circuit.

        Args:
            index (int, optional): The target index. Defaults to 0.
        """
        if index < 0:
            index = self.gate_length() + index

        assert index >= 0 and index < self.gate_length()
        gate, qidx = self._gates.pop(index)

        return gate.copy() & qidx

    def adjust(self, index: int, reassigned_qubits: Union[int, list], is_adjust_value: bool = False):
        """ Adjust the placement for target CompositeGate/BasicGate/Operator.

        Args:
            index (int): The target Quantum Gate's index, **Start from 0**.
            reassigned_qubits (Union[int, list]): The new assigned qubits of target Quantum Gate
            is_adjust_value (bool): Whether the reassigned_qubits means the new qubit indexes or the adjustment
                value from original indexes.
        """
        if index < 0:
            index = self.gate_length() + index
        assert index >= 0 and index < self.gate_length()
        origin_gate, origin_qidx, origin_size = self._gates[index]

        if is_adjust_value:
            new_qubits = [v + reassigned_qubits for v in origin_qidx] if isinstance(reassigned_qubits, int) else \
                [v + reassigned_qubits[idx] for idx, v in enumerate(origin_qidx)]
        else:
            new_qubits = [reassigned_qubits] if isinstance(reassigned_qubits, int) else reassigned_qubits

        assert len(origin_qidx) == len(new_qubits)
        self._gates[index] = (origin_gate, new_qubits, origin_size)

    ####################################################################
    ############            CompositeGate Utils             ############
    ####################################################################
    def inverse(self) -> CompositeGate:
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        _gates = CompositeGate()
        _gates._gates = self._gates.inverse()

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
        matrix_width = self.width() if local else max(self.qubits) + 1

        circuit_matrix = CircuitMatrix(device, self._precision)
        assigned_gates = self.flatten_gates(True) if not local else self._get_local_gates()

        return circuit_matrix.get_unitary_matrix(assigned_gates, matrix_width)

    def _get_local_gates(self) -> list:
        local_qidx_mapping = {}
        for i, qidx in enumerate(self.qubits):
            local_qidx_mapping[qidx] = i

        local_gates = []
        flatten_gates = self.gate_decomposition(self_flatten=False)
        for gate, qidx, _ in flatten_gates:
            related_qidx = [local_qidx_mapping[q] for q in qidx]
            lgate = gate.copy() & related_qidx
            local_gates.append(lgate)

        return local_gates
