#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 9:41
# @Author  : Han Yu, Kaiqi Li
# @File    : circuit.py
from __future__ import annotations

from typing import Union, List
import numpy as np
import random

from QuICT.core.qubit import Qubit, Qureg
from QuICT.core.layout import Layout, SupremacyLayout
from QuICT.core.gate import BasicGate, H, Measure, gate_builder, CompositeGate
from QuICT.core.utils import (
    GateType,
    CircuitBased,
    CircuitMatrix,
    unique_id_generator
)
from QuICT.core.operator import (
    CheckPoint,
    NoiseGate,
    Operator,
    CheckPointChild
)
from .dag_circuit import DAGCircuit

from QuICT.tools import Logger
from QuICT.tools.exception.core import *


class Circuit(CircuitBased):
    """ Implement a quantum circuit

    Circuit is the core part of the framework.

    Attributes:
        wires(Union[Qureg, int]): the number of qubits for the circuit.
        name(str): the name of the circuit.
        topology(list<tuple<int, int>>):
            The topology of the circuit. When the topology list is empty, it will be seemed as fully connected.
        fidelity(float): the fidelity of the circuit
        ancilla_qubits(list<int>): The indexes of ancilla qubits for current circuit.
    """
    @property
    def qubits(self) -> Qureg:
        return self._qubits

    @property
    def ancilla_qubits(self) -> List[int]:
        return self._ancillae_qubits

    @ancilla_qubits.setter
    def ancilla_qubits(self, ancilla_qubits: List[int]):
        for idx in ancilla_qubits:
            if idx < 0 or idx >= self.width():
                raise IndexExceedError(
                    "circuit.ancilla_qubits", [0, self.width()], idx
                )

            self._ancillae_qubits.append(idx)

    @property
    def topology(self) -> Layout:
        return self._topology

    @topology.setter
    def topology(self, topology: Layout):
        assert isinstance(topology, Layout), TypeError("Circuit.topology", "Layout", type(topology))
        assert topology.qubit_number == self.width(), ValueError(
            "Circuit.topology.qubit_number", self.width(), topology.qubit_number
        )

        self._topology = topology

    def __init__(
        self,
        wires,
        name: str = None,
        topology: Layout = None,
        ancilla_qubits: List[int] = None
    ):
        if name is None:
            name = "circuit_" + unique_id_generator()

        super().__init__(name)
        self._ancillae_qubits = []
        self._topology = None
        self._checkpoints = []
        self._logger = Logger("circuit")

        if isinstance(wires, Qureg):
            self._qubits = wires
        else:
            self._qubits = Qureg(wires)

        if ancilla_qubits is not None:
            assert max(ancilla_qubits) < self.width()
            self.ancilla_qubits = ancilla_qubits

        self._logger.debug(f"Initial Quantum Circuit {name} with {len(self._qubits)} qubits.")
        if topology is not None:
            self.topology = topology
            self._logger.debug(f"The Layout for Quantum Circuit is {self._topology}.")

    def __del__(self):
        """ release the memory """
        self._gates = None
        self._qubits = None
        self._topology = None

    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "circuit | circuit"
        to add the gate of circuit into the circuit/qureg/qubit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
        Raise:
            TypeError: the type of targets is wrong
        """
        assert isinstance(targets, Circuit), TypeError("Circuit.or", "Circuit", type(targets))

        targets.extend(self)

    ####################################################################
    ############         Circuit Qubits Operators           ############
    ####################################################################
    def __call__(self, indexes: object):
        """ get a smaller qureg from this circuit

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) list<int>
                3) Qubit
                4) Qureg
        Returns:
            Qureg: the qureg correspond to the indexes
        Exceptions:
            TypeError: the type of indexes is error.
        """
        if isinstance(indexes, (Qubit, Qureg)):
            indexes = self.qubits.index(indexes)

        if isinstance(indexes, int):
            indexes = [indexes]

        if not isinstance(indexes, list):
            raise TypeError(
                "Circuit.call", "int/list[int]/Qubit/Qureg", type(indexes)
            )

        self._pointer = indexes
        return self

    def __getitem__(self, item):
        """ to fit the slice operator, overloaded this function.

        get a smaller qureg/qubit from this circuit

        Args:
            item(int/slice): slice passed in.
        Return:
            Qubit/Qureg: the result or slice
        """
        return self.qubits[item]

    def add_qubit(self, qubits: Union[Qureg, Qubit, int], is_ancillary_qubit: bool = False):
        """ add additional qubits in circuit.

        Args:
            qubits Union[Qureg, Qubit, int]: The new qubits, if it is int, means the number of new qubits.
            is_ancillae_qubit (bool, optional): whether the given qubits is ancillae, default to False.
        """
        assert isinstance(qubits, (Qureg, Qubit, int)), \
            TypeError("Circuit.add_qubit", "[Qureg, Qubit, int]", type(qubits))
        if isinstance(qubits, int):
            assert qubits > 0, IndexExceedError("Circuit.add_qubit", ">= 0", {qubits})
            qubits = Qureg(qubits)
        elif isinstance(qubits, Qubit):
            qubits = Qureg(qubits)

        self._qubits = self._qubits + qubits
        if is_ancillary_qubit:
            self._ancillae_qubits += list(range(self.width() - len(qubits), self.width()))

        # self._logger.debug(f"Quantum Circuit {self._name} add {len(qubits)} qubits.")

    def reset_qubits(self):
        """ Reset all qubits in current circuit. """
        self._qubits.reset_qubits()
        self._logger.debug(f"Reset qubits' measured result in the Quantum Circuit {self._name}.")

    ####################################################################
    ############          Circuit Gates Operators           ############
    ####################################################################
    def find_position(self, cp_child: CheckPointChild):
        position = -1
        if cp_child is None:
            return position

        for cp in self._checkpoints:
            if cp.uid == cp_child.uid:
                position = cp.position
                cp.position = cp_child.shift
            elif position != -1:
                # change the related position for backward checkpoint
                cp.position = cp_child.shift

        return position

    def get_DAG_circuit(self) -> DAGCircuit:
        """
        Translate a quantum circuit to a directed acyclic graph
        via quantum gates dependencies (The commutation of quantum gates).

        The nodes in the graph represented the quantum gates, and the edges means the two quantum
        gates is non-commutation. In other words, a directed edge between node A with quantum gate GA
        and node B with quantum gate GB, the quantum gate GA does not commute with GB.

        The nodes in the graph have the following attributes:
        'name', 'gate', 'cargs', 'targs', 'qargs', 'successors', 'predecessors'.

        **Reference:**

        [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
        Exact and practical pattern matching for quantum circuit optimization.
        `arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

        Returns:
            DAGCircuit: A directed acyclic graph represent current quantum circuit
        """
        try:
            return DAGCircuit(self)
        except Exception as e:
            raise CircuitDAGError(e)

    ####################################################################
    ############          Circuit Build Operators           ############
    ####################################################################
    def extend(self, gates: Union[BasicGate, CompositeGate]):
        """ Add a CompositeGate/Circuit to the circuit.

        Args:
            gates (Union[BasicGate, CompositeGate]): the compositegate or circuit to be added to the circuit
        """
        assert isinstance(gates, (Circuit, CompositeGate)), \
            "The circuit extend method only accept CompositeGate or Circuit."

        if self._pointer is not None:
            gate_args = gates.width()
            assert gate_args == len(self._pointer), GateQubitAssignedError(
                f"{gates.name} need {gate_args} indexes, but given {len(self._pointer)}"
            )

            gate_qidxes = self._pointer[:]
        else:
            if isinstance(gates, CompositeGate):
                gate_qidxes = gates.qubits
            else:
                gate_qidxes = self._get_extend_circuit_qidxes(gates)

        if isinstance(gates, Circuit):
            gates = gates.to_compositegate()
            if gates.width() != len(gate_qidxes):
                gate_qidxes = [gate_qidxes[idx] for idx in gates.qubits]

            position = -1
        else:
            position = self.find_position(gates.checkpoint)

        if position == -1:
            self._gates.append((gates, gate_qidxes, gates.size()))
        else:
            self._gates.insert(position, (gates, gate_qidxes, gates.size()))

        self._pointer = None

    def append(self, op: Union[BasicGate, Operator]):
        """ Add a Quantum Gate or Operator into current circuit.

        Args:
            op (Union[BasicGate, Operator]): The Quantum Gate or Operator
        """
        if isinstance(op, BasicGate):
            self._add_gate(op)
        elif isinstance(op, Operator):
            self._add_operator(op)
        else:
            raise TypeError(
                "Circuit.append.gate", "BasicGate/Operator", {type(op)}
            )

        self._pointer = None

    def insert(self, gate: Union[CompositeGate, BasicGate], insert_idx: int):
        """ Insert a Quantum Gate into current Circuit, only support BasicGate/CompositeGate.

        Args:
            gate (Union[CompositeGate, BasicGate]): The quantum gate want to insert
            insert_idx (int): the index of insert
        """
        assert isinstance(gate, (BasicGate, Operator, CompositeGate)), \
            TypeError("CompositeGate.insert", "BasicGate", type(gate))
        gate_args = gate.qubits if isinstance(gate, CompositeGate) else gate.cargs + gate.targs
        gate_size = gate.size() if isinstance(gate, CompositeGate) else 1
        if len(gate_args) == 0:
            raise GateQubitAssignedError(f"{gate.type} need qubit indexes to insert into Composite Gate.")

        for garg in gate_args:
            assert garg >= 0 and garg < self.width(), \
                GateQubitAssignedError(f"Gate's assigned qubits should within [0, {self.width()}]")

        self._gates.insert(insert_idx, (gate, gate_args, gate_size))

    def _add_gate(self, gate: BasicGate):
        """ add a quantum gate into circuit.

        Args:
            gate(BasicGate): The Quantum Gate want to append in current circuit
        """
        if self._pointer is not None:
            gate_args = gate.controls + gate.targets
            assert len(self._pointer) == gate_args, \
                GateQubitAssignedError(f"{gate.type} need {gate_args} indexes, but given {len(self._pointer)}")

            qubit_index = self._pointer[:]
        else:
            gate_qargs = gate.cargs + gate.targs
            gate_aqubits = gate.assigned_qubits
            if len(gate_qargs) == 0 and len(gate_aqubits) == 0:
                if gate.is_single():
                    self._add_gate_to_all_qubits(gate)
                    return
                elif gate.targets + gate.controls == self.width():
                    qubit_index = list(range(self.width()))
                else:
                    raise GateQubitAssignedError(f"{gate.type} need qubit indexes to add into Composite Gate.")
            else:
                qubit_index = gate_qargs if len(gate_qargs) > 0 else \
                    [self.qubits.index(aq) for aq in gate.assigned_qubits]

        self._gates.append((gate, qubit_index, 1))

    def _add_gate_to_all_qubits(self, gate: BasicGate):
        """ Add gate to all qubits.

        Args:
            gate (BasicGate): The quantum gate.
        """
        for idx in range(self.width()):
            if gate.variables > 0:
                self._gates.append((gate.copy(), [idx], 1))
            else:
                self._gates.append((gate, [idx], 1))

    def _add_operator(self, op: Operator):
        """ Add operator. """
        if isinstance(op, CheckPoint):
            self._checkpoints.append(op)
            return

        if self._pointer is not None:
            if len(self._pointer) != op.targets:
                raise CircuitAppendError("Failure to add Trigger into Circuit, as un-matched qureg.")

            op_qidxes = self._pointer[:]
        else:
            if not op.targs:
                raise CircuitAppendError("Operators need be assigned qubits before add into circuit.")

            for targ in op.targs:
                if targ >= self.width():
                    raise CircuitAppendError("The trigger's target exceed the width of the circuit.")

            op_qidxes = op.targs

        size = 1 if isinstance(op, NoiseGate) else 0
        self._gates.append((op, op_qidxes, size))
        self._logger.debug(f"Add an Operator {type(op)} with qubit indexes {op_qidxes}.")

    def _get_extend_circuit_qidxes(self, circuit: Circuit):
        """ Append qubits from expand circuit. """
        ec_qidxes = []
        aqubit_idxes = circuit.ancilla_qubits
        for idx, qubit in enumerate(circuit.qubits):
            if qubit in self.qubits:
                ec_qidxes.append(self._qubits.index(qubit))
            else:
                is_ancilla = True if idx in aqubit_idxes else False
                self.add_qubit(qubit, is_ancilla)
                ec_qidxes.append(self.width() - 1)

        return ec_qidxes

    def random_append(
        self,
        rand_size: int = 10,
        typelist: list = None,
        random_params: bool = False,
        probabilities: list = None
    ):
        """ add some random gate to the circuit, not include Unitary, Permutation and Permutation_FX Gate.

        Args:
            rand_size(int): the number of the gate added to the circuit
            typelist(list<GateType>): the type of gate, default contains
                Rx, Ry, Rz, Cx, Cy, Cz, CRz, Ch, Rxx, Ryy, Rzz and FSim
            random_params(bool): whether using random parameters for all quantum gates with parameters.
            probabilities: The probability of append for each gates
        """
        if typelist is None:
            typelist = [
                GateType.rx, GateType.ry, GateType.rz,
                GateType.cx, GateType.cy, GateType.crz,
                GateType.ch, GateType.cz, GateType.rxx,
                GateType.ryy, GateType.rzz, GateType.fsim
            ]

        unsupported_gate_type = [GateType.unitary, GateType.perm, GateType.perm_fx]
        if len(set(typelist) & set(unsupported_gate_type)) != 0:
            raise CircuitSpecialAppendError(
                f"{set(typelist) & set(unsupported_gate_type)} is not support in random append."
            )

        if probabilities is not None:
            if not np.isclose(sum(probabilities), 1, atol=1e-6):
                raise ValueError("Circuit.random_append.probabilities", "sum to 1", sum(probabilities))

            if len(probabilities) != len(typelist):
                raise CircuitSpecialAppendError(
                    "The length of probabilities should equal to the length of Gate Typelist."
                )

        self._logger.debug(f"Random append {rand_size} quantum gates from {typelist} with probability {probabilities}.")
        gate_prob = probabilities
        gate_indexes = list(range(len(typelist)))
        for _ in range(rand_size):
            gate_type = typelist[np.random.choice(gate_indexes, p=gate_prob)]
            r_gate = gate_builder(gate_type, random_params=random_params)
            random_assigned_qubits = random.sample(range(self.width()), r_gate.controls + r_gate.targets)

            self._gates.append((r_gate, random_assigned_qubits, 1))

    def supremacy_append(self, repeat: int = 1, pattern: str = "ABCDCDAB"):
        """
        Add a supremacy circuit to the circuit

        Args:
            repeat(int): the number of two-qubit gates' sequence
            pattern(str): indicate the two-qubit gates' sequence
        """
        qubits = len(self.qubits)
        supremacy_layout = SupremacyLayout(qubits)
        supremacy_typelist = [GateType.sx, GateType.sy, GateType.sw]
        self._logger.debug(
            f"Append Supremacy Circuit with mapping pattern sequence {pattern} and repeat {repeat} times."
        )

        self._add_gate_to_all_qubits(H)

        for i in range(repeat * len(pattern)):
            for q in range(qubits):
                gate_type = supremacy_typelist[np.random.randint(0, 3)]
                fgate = gate_builder(gate_type)
                self._gates.append((fgate, [q], 1))

            current_pattern = pattern[i % (len(pattern))]
            if current_pattern not in "ABCD":
                raise ValueError("Circuit.append_supremacy.pattern", "[A, B, C, D]", current_pattern)

            edges = supremacy_layout.get_edges_by_pattern(current_pattern)
            for e in edges:
                gate_params = [np.pi / 2, np.pi / 6]
                gate_args = [int(e[0]), int(e[1])]
                fgate = gate_builder(GateType.fsim, params=gate_params)

                self._gates.append((fgate, gate_args, 1))

        self._add_gate_to_all_qubits(Measure)

    ####################################################################
    ############                Circuit Utils               ############
    ####################################################################
    def to_compositegate(self) -> CompositeGate:
        """ Get CompositeGate from current Circuit. """
        _cgate = CompositeGate()
        for gate in self.gates:
            gate | _cgate

        return _cgate

    def inverse(self) -> Circuit:
        """ the inverse of Circuit

        Returns:
            Circuit: the inverse of the gateSet
        """
        _cir = Circuit(self.width())
        inverse_gates = []
        for gate, indexes, size in self._gates[::-1]:
            if not isinstance(gate, Operator):
                inverse_gates.append((gate.inverse(), indexes, size))
            else:
                inverse_gates.append((gate, indexes, size))

        _cir._gates = inverse_gates

        return _cir

    def matrix(self, device: str = "CPU") -> np.ndarray:
        """ Generate the circuit's unitary matrix which compose by all quantum gates' matrix in current circuit.

        Args:
            device (str, optional): The device type for generate circuit's matrix, one of [CPU, GPU]. Defaults to "CPU".
        """
        assert device in ["CPU", "GPU"]
        circuit_matrix = CircuitMatrix(device, self._precision)

        return circuit_matrix.get_unitary_matrix(self.flatten_gates(True), self.width())

    def sub_circuit(
        self,
        start: int = 0,
        max_size: int = -1,
        qubit_limit: Union[int, List[int], Qureg] = [],
        gate_limit: List[GateType] = []
    ):
        """ Get a sub circuit from current circuit with target gate and qubit limitation. Operators will be ignore.

        Args:
            start(int): the start gate's index, default 0
            max_size(int): max size of the sub circuit, default -1 without limit
            qubit_limit(int/list<int>/Qubit/Qureg): the required qubits' indexes, if [], accept all qubits.
                default to be [].
            gate_limit(List[GateType]): list of required gate's type, if [], accept all quantum gate. default to be [].
        Return:
            Circuit: the sub circuit
        """
        max_size_for_logger = self.size() if max_size == -1 else max_size
        self._logger.debug(
            f"Get {max_size_for_logger} gates from gate index {start}" +
            f" with target qubits {qubit_limit} and gate limit {gate_limit}."
        )
        if qubit_limit:
            if isinstance(qubit_limit, Qureg):
                qubit_limit = [self._qubits.index(qubit) for qubit in qubit_limit]
            elif isinstance(qubit_limit, Qubit):
                qubit_limit = [self._qubits.index(qubit_limit)]
            elif isinstance(qubit_limit, int):
                qubit_limit = [qubit_limit]

            for target in qubit_limit:
                if target < 0 or target >= self.width():
                    raise IndexExceedError("Circuit.sub_circuit.qubit_limit", [0, self.width()], target)

            set_tqubits = set(qubit_limit)

        sub_circuit = Circuit(self.width()) if not qubit_limit else Circuit(len(qubit_limit))
        flatten_gates = self.flatten_gates()
        temp_size = 0
        for gate_index in range(start, len(flatten_gates)):
            gate = flatten_gates[gate_index]
            if isinstance(gate, Operator):
                continue

            qidxes = gate.cargs + gate.targs
            gate_args = set(qidxes)
            is_append_in_subc = True
            if (qubit_limit and gate_args & set(set_tqubits) != gate_args):
                is_append_in_subc = False

            if (gate_limit and gate.type not in gate_limit):
                is_append_in_subc = False

            if is_append_in_subc:
                new_qidxes = [qubit_limit.index(q) for q in qidxes] if qubit_limit else qidxes
                gate | sub_circuit(new_qidxes)
                temp_size += 1

            if max_size != -1 and temp_size >= max_size:
                break

        return sub_circuit
