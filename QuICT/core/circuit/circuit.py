#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 9:41
# @Author  : Han Yu, Kaiqi Li
# @File    : circuit.py
from typing import Union
import numpy as np

from QuICT.core.qubit import Qubit, Qureg
from QuICT.core.exception import TypeException
from QuICT.core.layout import Layout, SupremacyLayout
from QuICT.core.gate import BasicGate, H, Measure, build_random_gate, build_gate
from QuICT.core.utils import (
    GateType,
    CircuitInformation,
    matrix_product_to_circuit
)
from QuICT.core.operator import Trigger, CheckPoint, Operator, CheckPointChild


# global circuit id count
circuit_id = 0


class Circuit(object):
    """ Implement a quantum circuit

    Circuit is the core part of the framework.

    Attributes:
        id(int): the unique identity code of a circuit, which is generated globally.
        name(str): the name of the circuit
        qubits(Qureg): the qureg formed by all qubits of the circuit
        gates(list<BasicGate>): all gates attached to the circuit
        topology(list<tuple<int, int>>):
            The topology of the circuit. When the topology list is empty, it will be seemed as fully connected.
        fidelity(float): the fidelity of the circuit

    Private Attributes:
        _idmap(dictionary): the map from qubit's id to its index in the circuit
    """

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> int:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def qubits(self) -> Qureg:
        return self._qubits

    @qubits.setter
    def qubits(self, qubits: Qureg):
        self._qubits = qubits

    @property
    def gates(self) -> list:
        return self._gates

    @gates.setter
    def gates(self, gates):
        self._gates = gates

    @property
    def topology(self) -> Layout:
        return self._topology

    @topology.setter
    def topology(self, topology: Layout):
        if topology is None:
            self._topology = None
            return

        if not isinstance(topology, Layout):
            raise TypeError("Only support Layout as circuit topology.")

        if topology.qubit_number != self.width():
            raise ValueError(f"The qubit number is not mapping. {topology.qubit_number}")

        self._topology = topology

    @property
    def fidelity(self) -> float:
        return self._fidelity

    @fidelity.setter
    def fidelity(self, fidelity):
        if fidelity is None:
            self._fidelity = None
            return

        if not isinstance(fidelity, float) or fidelity < 0 or fidelity > 1.0:
            raise Exception("fidelity should be in [0, 1]")

        self._fidelity = fidelity

    def __init__(
        self,
        wires,
        name: str = None,
        topology: Layout = None,
        fidelity: float = None
    ):
        """
        generator a circuit

        Args:
            wires(int/qureg/[qubit]): the number of qubits in the circuit
        """
        global circuit_id
        self._id = circuit_id
        circuit_id = circuit_id + 1
        self._name = "circuit_" + str(self.id) if name is None else name
        self._topology = topology
        self._fidelity = fidelity
        self._gates = []
        self._checkpoints = []
        self._pointer = None

        if isinstance(wires, Qureg):
            self._qubits = wires
        else:
            self._qubits = Qureg(wires)

        self._update_idmap()

    def _update_idmap(self):
        self._idmap = {}
        for idx, qubit in enumerate(self.qubits):
            self._idmap[qubit.id] = idx

    def __del__(self):
        """ release the memory """
        self.gates = None
        self.qubits = None
        self.topology = None
        self.fidelity = None

    def width(self):
        """ the number of qubits in circuit

        Returns:
            int: the number of qubits in circuit
        """
        return len(self.qubits)

    def size(self):
        """ the size of the circuit

        Returns:
            int: the number of gates in circuit
        """
        return len(self.gates)

    def count_2qubit_gate(self):
        """ the number of the two qubit gates in the circuit

        Returns:
            int: the number of the two qubit gates in the circuit
        """
        return CircuitInformation.count_2qubit_gate(self.gates)

    def count_1qubit_gate(self):
        """ the number of the one qubit gates in the circuit

        Returns:
            int: the number of the one qubit gates in the circuit
        """
        return CircuitInformation.count_1qubit_gate(self.gates)

    def count_gate_by_gatetype(self, gate_type):
        """ the number of the gates which are some type in the circuit

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type in the circuit
        """
        return CircuitInformation.count_gate_by_gatetype(self.gates, gate_type)

    def depth(self):
        """ the depth of the circuit.

        Returns:
            int: the depth of the circuit
        """
        return CircuitInformation.depth(self.gates, self.width())

    def __str__(self):
        circuit_info = {
            "name": self.name,
            "width": self.width(),
            "size": self.size(),
            "depth": self.depth(),
            "1-qubit gates": self.count_1qubit_gate(),
            "2-qubit gates": self.count_2qubit_gate()
        }

        return str(circuit_info)

    def draw(self, method='matp', filename=None):
        """ draw the photo of circuit in the run directory

        Args:
            filename(str): the output filename without file extensions,
                           default to be the name of the circuit
            method(str): the method to draw the circuit
                matp: matplotlib
                command : command
                tex : tex source
        """
        from QuICT.tools.drawer import PhotoDrawer, TextDrawing

        if method == 'matp':
            if filename is None:
                filename = str(self.id) + '.jpg'
            elif '.' not in filename:
                filename += '.jpg'

            photoDrawer = PhotoDrawer()
            photoDrawer.run(self, filename)
        elif method == 'command':
            textDrawing = TextDrawing([i for i in range(len(self.qubits))], self.gates)
            if filename is None:
                print(textDrawing.single_string())
                return
            elif '.' not in filename:
                filename += '.txt'

            textDrawing.dump(filename)

    def qasm(self):
        """ get OpenQASM 2.0 describe for the circuit

        Returns:
            str: OpenQASM 2.0 describe or "error" when there are some gates cannot be resolved
        """
        qreg = len(self.qubits)
        creg = self.count_gate_by_gatetype(GateType.measure)

        return CircuitInformation.qasm(qreg, creg, self.gates)

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
        if isinstance(indexes, int) or type(indexes) is list:
            indexes = self.qubits(indexes)

        if isinstance(indexes, Qubit):
            indexes = Qureg(indexes)

        if not isinstance(indexes, Qureg):
            raise TypeError("only accept int/list[int]/Qubit/Qureg")

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
        if not isinstance(targets, Circuit):
            raise TypeError("Only support circuit | circuit.")

        if not self.qubits == targets.qubits:
            diff_qubits = targets.qubits.diff(self.qubits)
            targets.update_qubit(diff_qubits, is_append=True)

        targets.extend(self.gates)

    def update_qubit(self, qubits: Qureg, is_append: bool = False):
        """ Update the qubits in circuit.

        Args:
            qubits (Qureg): The new qubits.
            is_append (bool, optional): whether add qubits or replace qubits. Defaults to False, add qubits.
        """
        if not is_append:
            self._qubits = qubits
            self._idmap = {}
        else:
            self._qubits = self._qubits + qubits

        for idx, qubit in enumerate(self.qubits):
            self._idmap[qubit.id] = idx

    def replace_gate(self, idx: int, gate: BasicGate):
        assert idx >= 0 and idx < len(self._gates), "The index of replaced gate is wrong."
        assert isinstance(gate, BasicGate), "The replaced gate must be a quantum gate or noised gate."
        self._gates[idx] = gate

    def extend(self, gates: list):
        """ add gates to the circuit

        Args:
            gates(list<BasicGate>): the gate to be added to the circuit
        """
        position = -1
        if not isinstance(gates, list):
            gate_cp: CheckPointChild = gates.checkpoint
            if gate_cp is not None:
                position = gate_cp.find_position(self._checkpoints)

            gates = gates.gates
            
        for gate in gates:
            if position == -1:
                self.append(gate, is_extend=True)
            else:
                self.append(gate, is_extend=True, insert_idx=position)
                position += 1

        self._pointer = None

    def append(self, op: Union[BasicGate, Operator], is_extend: bool = False, insert_idx: int = -1):
        qureg = self._pointer[:] if self._pointer else None
        if not is_extend:
            self._pointer = None

        if isinstance(op, BasicGate):
            self._add_gate(op, qureg, insert_idx)
        elif isinstance(op, Trigger):
            self._add_trigger(op, qureg)
        elif isinstance(op, CheckPoint):
            self._checkpoints.append(op)
        elif isinstance(op, Operator):
            self._gates.append(op)
        else:
            raise TypeError(f"Circuit can append a Trigger/BasicGate/NoiseGate, not {type(op)}.")

    def _add_gate(self, gate: BasicGate, qureg: Qureg, insert_idx: int):
        """ add a gate into some qureg

        Args:
            gate(BasicGate)
            qureg(Qureg)
        """
        args_num = gate.controls + gate.targets
        gate_ctargs = gate.cargs + gate.targs
        is_assigned = gate.assigned_qubits or gate_ctargs
        if not qureg:
            if not is_assigned:
                if gate.is_single():
                    self._add_gate_to_all_qubits(gate)
                    return
                elif args_num == self.width():
                    qureg = self.qubits
                else:
                    raise KeyError(f"{gate.type} need assign qubits to add into circuit.")
            else:
                qureg = self.qubits[gate_ctargs] if gate_ctargs else gate.assigned_qubits

        if len(qureg) > args_num:
            qureg = qureg[gate_ctargs]

        gate = gate.copy()
        gate.cargs = [self._idmap[qureg[idx].id] for idx in range(gate.controls)]
        gate.targs = [self._idmap[qureg[idx].id] for idx in range(gate.controls, gate.controls + gate.targets)]
        gate.assigned_qubits = qureg
        gate.update_name(qureg[0].id, len(self.gates))

        if insert_idx == -1:
            self.gates.append(gate)
        else:
            self.gates.insert(insert_idx, gate)

    def _add_gate_to_all_qubits(self, gate):
        for idx in range(self.width()):
            new_gate = gate.copy()
            new_gate.targs = [idx]
            new_gate.assigned_qubits = self.qubits(idx)
            new_gate.update_name(self.qubits[idx].id, len(self.gates))

            self.gates.append(new_gate)

    def _add_trigger(self, op: Trigger, qureg: Qureg):
        if qureg:
            assert len(qureg) == op.targets
            op.targs = [self._idmap[qureg[idx].id] for idx in range(op.targets)]
        else:
            if not op.targs:
                raise KeyError("Trigger need assign qubits to add into circuit.")

            for targ in op.targs:
                assert targ < self.width(), "The trigger's target exceed the width of the circuit."

        self.gates.append(op)

    def sub_circuit(
        self,
        targets,
        start: int = 0,
        max_size: int = -1,
        remove: bool = False
    ):
        """ get a sub circuit

        Args:
            targets(int/list<int>/Qureg): target qubits indexes.
            start(int): the start gate's index, default 0
            max_size(int): max size of the sub circuit, default -1 without limit
            remove(bool): whether deleting the slice gates from origin circuit, default False
        Return:
            Circuit: the sub circuit
        """
        if isinstance(targets, Qureg):
            targets = [self.index_for_qubit(qubit) for qubit in targets]
        elif isinstance(targets, int):
            targets = [targets]

        # the mapping from circuit's index to sub-circuit's index
        circuit_width = self.width()
        targets_mapping = [0] * circuit_width
        for idx, target in enumerate(targets):
            if target < 0 or target >= circuit_width:
                raise Exception('list index out of range')
            targets_mapping[target] = idx

        # build sub_circuit
        sub_circuit = Circuit(len(targets))
        set_targets = set(targets)
        targets_gates = self.gates_for_qubit(self.qubits(targets))
        sub_gates = []

        for gate_index in range(start, len(targets_gates)):
            gate = targets_gates[gate_index]
            gate_args = set(gate.cargs + gate.targs)
            if gate_args & set_targets == gate_args:
                _gate = gate.copy()
                _gate.targs = [targets_mapping[targ] for targ in _gate.targs]
                _gate.cargs = [targets_mapping[carg] for carg in _gate.cargs]
                sub_gates.append(_gate)

                if remove:
                    self.gates.remove(gate)

            if len(sub_gates) >= max_size and max_size != -1:
                break

        if remove:
            self._update_gate_index()

        sub_circuit.extend(sub_gates)
        return sub_circuit

    def _update_gate_index(self):
        for index, gate in enumerate(self.gates):
            gate_type, gate_qb, gate_idx = gate.name.split('-')

            if int(gate_idx) != index:
                gate.name = '-'.join([gate_type, gate_qb, str(index)])

    def index_for_qubit(self, qubit, ancilla=None) -> int:
        """ find the index of qubit in this circuit

        the index ignored the ancilla qubit

        Args:
            qubit(Qubit): the qubit need to be indexed.
            ancilla(Qureg): the ancillary qubit

        Returns:
            int: the index of the qubit.

        Raises:
            Exception: the qubit is not in the circuit
        """
        if not isinstance(qubit, Qubit):
            raise TypeError(f"Only support qubit here, not {type(qubit)}.")

        if qubit.id not in self._idmap.keys():
            raise Exception("the qubit is not in the circuit or it is an ancillary qubit.")

        if ancilla is None:
            return self._idmap[qubit.id]

        if not isinstance(ancilla, Qureg):
            raise TypeError(f"Ancilla must be a Qureg here, not {type(ancilla)}.")

        enterspace = 0
        for q in self.qubits:
            if q not in ancilla:
                enterspace += 1

            if q.id == qubit.id:
                return enterspace

    def gates_for_qubit(self, qubits: Qureg) -> list:
        """ Return the gates for given qubits

        Args:
            qubits (Qureg/Qubit): The target qubits.

        Returns:
            list: The list of gates
        """
        if isinstance(qubits, Qureg):
            qubits = [self.index_for_qubit(qubit) for qubit in qubits]
        elif isinstance(qubits, Qubit):
            qubits = [self.index_for_qubit(qubits)]
        else:
            raise TypeError(f"Only supports use Qubit/Qureg to find the gates. {type(qubits)}")

        set_qubits = set(qubits)
        q_gates = []
        for gate in self.gates:
            ctargs = set(gate.cargs + gate.targs)
            if set_qubits & ctargs:
                q_gates.append(gate)

        return q_gates

    def random_append(
        self,
        rand_size: int = 10,
        typelist: list = None,
        random_params: bool = False
    ):
        """ add some random gate to the circuit

        Args:
            rand_size(int): the number of the gate added to the circuit
            typelist(list<GateType>): the type of gate, default contains
                Rx, Ry, Rz, Cx, Cy, Cz, CRz, Ch, Rxx, Ryy, Rzz and FSim
        """
        if typelist is None:
            typelist = [
                GateType.rx, GateType.ry, GateType.rz,
                GateType.cx, GateType.cy, GateType.crz,
                GateType.ch, GateType.cz, GateType.Rxx,
                GateType.Ryy, GateType.Rzz, GateType.fsim
            ]

        n_qubit = self.width()
        for _ in range(rand_size):
            rand_type = np.random.randint(0, len(typelist))
            gate_type = typelist[rand_type]
            self.append(build_random_gate(gate_type, n_qubit, random_params))

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

        self._add_gate_to_all_qubits(H)

        for i in range(repeat * len(pattern)):
            for q in range(qubits):
                gate_type = supremacy_typelist[np.random.randint(0, 3)]
                self.append(build_gate(gate_type, q))

            current_pattern = pattern[i % (len(pattern))]
            if current_pattern not in "ABCD":
                raise KeyError(f"Unsupported pattern {pattern[i]}, please use one of 'A', 'B', 'C', 'D'.")

            edges = supremacy_layout.get_edges_by_pattern(current_pattern)
            for e in edges:
                gate_params = [np.pi / 2, np.pi / 6]
                gate_args = [int(e[0]), int(e[1])]
                fgate = build_gate(GateType.fsim, gate_args, gate_params)

                self.append(fgate)

        self._add_gate_to_all_qubits(Measure)

    def matrix_product_to_circuit(self, gate) -> np.ndarray:
        """ extend a gate's matrix in the all circuit unitary linear space

        gate's matrix tensor products some identity matrix.

        Args:
            gate(BasicGate): the gate to be extended.
        """
        return matrix_product_to_circuit(gate.matrix, gate.cargs + gate.targs, len(self.qubits))

    def remapping(self, qureg: Qureg, mapping: list, circuit_update: bool = False):
        """ Realignment the qubits by the given mapping.

        Args:
            qureg (Qureg): The qubits which need to permutate.
            mapping (list): The order of permutation.
            circuit_update (bool, optional): whether rearrange the qubits in circuit. Defaults to False.
        """
        if not isinstance(qureg, Qureg):
            raise TypeException("Qureg Only.", qureg)

        if len(qureg) != len(mapping):
            raise ValueError(f"the length of mapping {len(mapping)} must equal to the qubits' number {len(qureg)}.")

        current_index = [self.index_for_qubit(qubit) for qubit in qureg]
        remapping_index = [current_index[m] for m in mapping]
        remapping_qureg = self.qubits[remapping_index]

        if circuit_update:
            self.update_qubit(remapping_qureg, is_append=False)

        qureg[:] = remapping_qureg
