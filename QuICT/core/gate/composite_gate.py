#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
from __future__ import annotations

from typing import Union, List
import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.operator import CheckPointChild
from QuICT.core.utils import CircuitBased, CircuitMatrix, CGATE_LIST, unique_id_generator
from QuICT.tools.exception.core import ValueError, CompositeGateAppendError, TypeError, GateQubitAssignedError


class CompositeGate(CircuitBased):
    """ Implement a group of gate """
    @property
    def qubits(self) -> list:
        return sorted(self._qubits)

    @property
    def checkpoint(self):
        return self._check_point

    def __init__(self, name: str = None, gates: List[BasicGate, CompositeGate] = None):
        """
        Args:
            name (str, optional): the name of the composite gate. Defaults to None.
            gates (List[BasicGate, CompositeGate], optional): gates within this composite gate. Defaults to None.
        """
        if name is None:
            name = "composite_gate_" + unique_id_generator()

        super().__init__(name)
        self._check_point = None        # required checkpoint
        self._qubits = []

        if gates is not None:
            for gate in gates:
                if isinstance(gate, CompositeGate):
                    self.extend(gate)
                else:
                    self.append(gate)

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

        if CGATE_LIST:
            CGATE_LIST[-1].extend(self)

        return self

    def _mapping(self, targets: list):
        """ remapping the gates' target qubits

        Args:
            targets(list): the related qubits
        """
        qidx_mapping = {}
        for i, q in enumerate(self.qubits):
            qidx_mapping[q] = targets[i]

        new_gates = []
        for gate, qidxes, size in self._gates:
            new_q = [qidx_mapping[qidx] for qidx in qidxes]
            new_gates.append((gate, new_q, size))

        self._qubits = targets
        self._gates = new_gates

    def _update_qubit_limit(self, indexes: list):
        for idx in indexes:
            assert idx >= 0 and isinstance(idx, int)
            if idx not in self._qubits:
                self._qubits.append(idx)

    def _update_qubits_after_remove(self, indexes: list):
        """ Update qubits if remove any gate in current CompositeGate.

        Args:
            indexes (list): The original qubit indexes from removed Quantum Gate.
        """
        for idx in indexes:
            if not self._exist_qubits(idx):
                self._qubits.remove(idx)

    def _exist_qubits(self, idx: int) -> bool:
        """ Whether exist Quantum Gate with the given qubit indexes. """
        if idx not in self._qubits:
            return False

        for _, qidx, _ in self._gates:
            if idx in qidx:
                return True

        return False

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
        gate, qidxes, _ = self._gates[item]

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

        self._gates.append((gates, gate_qidxes, gates.size()))
        self._update_qubit_limit(gate_qidxes)
        self._pointer = None

    def append(self, gate: BasicGate):
        """ Add a quantum gate to current CompositeGate.

        Args:
            gate (BasicGate): The quantum gate need to append
        """
        if isinstance(gate, BasicGate):
            self._append_gate(gate)
        elif isinstance(gate, CheckPointChild):
            self._check_point = gate
        else:
            raise TypeError("CompositeGate.append", "BasicGate/CheckPoint", type(gate))

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

        self._update_qubit_limit(gate_args)
        self._gates.insert(insert_idx, (gate, gate_args, gate_size))

    def pop(self, index: int = -1):
        """ Pop the BasicGate/Operator/CompositeGate from current Quantum Circuit.

        Args:
            index (int, optional): The target index. Defaults to 0.
        """
        if index < 0:
            index = self.gate_length() + index

        assert index >= 0 and index < self.gate_length()
        gate, qidx, _ = self._gates.pop(index)
        self._update_qubits_after_remove(qidx)

        return gate.copy() & qidx

    def adjust(self, index: int, reassigned_qubits: Union[int, list], is_adjust_value: bool = False):
        """ Adjust the placement for target CompositeGate/BasicGate/Operator.

        Args:
            index (int): The target Quantum Gate's index, **Start from 0**.
            reassigned_qubits (Union[int, list]): The new assigned qubits of target Quantum Gate
            is_adjust_vale (bool): Whether the reassigned_qubits means the new qubit indexes or the adjustment
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
        self._update_qubit_limit(new_qubits)
        self._update_qubits_after_remove(origin_qidx)

    def _append_gate(self, gate: BasicGate):
        """ Add a BasicGate into the current CompositeGate

        Args:
            gate (BasicGate): The BasicGate need to added
        """
        if self._pointer is not None:
            gate_args = gate.controls + gate.targets
            assert len(self._pointer) == gate_args, \
                GateQubitAssignedError(f"{gate.type} need {gate_args} indexes, but given {len(self._pointer)}")

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
    def inverse(self) -> CompositeGate:
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        _gates = CompositeGate()
        inverse_gates = [(gate.inverse(), indexes, size) for gate, indexes, size in self._gates[::-1]]
        _gates._gates = inverse_gates
        _gates._qubits = self.qubits

        return _gates

    def copy(self) -> CompositeGate:
        """ Copy current CompositeGate. """
        _gates = CompositeGate()
        _gates.name = self.name
        _gates._qubits = self.qubits
        copy_gates = [(gate.copy(), indexes, size) for gate, indexes, size in self._gates]
        _gates._gates = copy_gates

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
