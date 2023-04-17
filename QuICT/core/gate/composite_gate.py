#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
from __future__ import annotations

from typing import Union
import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.operator import CheckPointChild
from QuICT.core.utils import (
    GateType,
    CircuitBased,
    CircuitMatrix,
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
        return sorted(self._qubits)

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
            if isinstance(gate, CompositeGate):
                gate & new_q

            new_gates.append((gate, new_q, size))

        self._qubits = targets
        self._gates = new_gates

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
        gate, qidxes, _ = self._gates[item]

        return gate & qidxes

    def extend(self, gates: CompositeGate):
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

    def _append_gate(self, gate: BasicGate):
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
    def depth(self, depth_per_qubits: bool = False) -> int:
        """ the depth of the CompositeGate.

        Returns:
            int: the depth
        """
        depth = np.zeros(self.width(), dtype=int)
        for gate, targs, _ in self._gates:
            targs = [self.qubits.index(targ) for targ in targs]
            if isinstance(gate, CompositeGate):
                gdepth = gate.depth(True)
                for i, targ in enumerate(targs):
                    depth[targ] += gdepth[i]
            else:
                depth[targs] = np.max(depth[targs]) + 1

        return np.max(depth) if not depth_per_qubits else depth

    def inverse(self):
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        _gates = CompositeGate()
        inverse_gates = [(gate.inverse(), indexes, size) for gate, indexes, size in self._gates[::-1]]
        _gates._gates = inverse_gates
        _gates._qubits = self.qubits

        return _gates

    def copy(self):
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
            local: whether consider only about qubits or not

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
        flatten_gates = self.gate_decomposition(False)
        for gate, qidx, _ in flatten_gates:
            related_qidx = [local_qidx_mapping[q] for q in qidx]
            lgate = gate & related_qidx
            local_gates.append(lgate)

        return local_gates

    def qasm_gates_only(self, creg: int, cbits: int, target_qubits: list = None):
        qasm_string = ""
        if target_qubits is not None:
            qidx_mapping = {}
            for i, q in enumerate(self.qubits):
                qidx_mapping[q] = target_qubits[i]

        for gate, targs, _ in self._gates:
            if target_qubits is not None:
                targs = [qidx_mapping[targ] for targ in targs]

            if isinstance(gate, CompositeGate):
                qasm_string += gate.qasm_gates_only(creg, cbits, targs)
                continue

            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{targs}] -> c[{cbits}];\n"
                cbits += 1
                cbits = cbits % creg
            else:
                qasm_string += gate.qasm(targs)

        return qasm_string
