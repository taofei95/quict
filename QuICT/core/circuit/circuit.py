#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:41
# @Author  : Han Yu
# @File    : _circuit.py
import numpy as np

from QuICT.core.exception import TypeException
from QuICT.core.qubit import Qubit, Qureg
from QuICT.core.layout import Layout, SupremacyLayout
from QuICT.core import GateType, get_gate, get_n_args, SUPREMACY_GATE_SET
from QuICT.core.gate import H, FSim, Measure

from .circuit_computing import getRandomList, CircuitInformation


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
        __idmap(dictionary): the map from qubit's id to its index in the circuit
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
    def qubits(self, qubits):
        self._qubits = qubits

    @property
    def gates(self) -> list:
        return self._gates

    @gates.setter
    def gates(self, gates):
        self._gates = gates

    @property
    def topology(self) -> list:
        return self._topology

    @topology.setter
    def topology(self, topology: Layout):
        if topology is None:
            self._topology = None
            return

        if not isinstance(topology, Layout):
            raise TypeError("Only support Layout as circuit topology.")

        if topology.qubit_number != self.circuit_width():
            raise ValueError(f"The qubit number is not mapping. {topology.qubit_number}")

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

        if isinstance(wires, int) or isinstance(wires, list):
            self._qubits = Qureg(wires)
        elif isinstance(wires, Qureg):
            self._qubits = wires
        elif isinstance(wires, Qubit):
            self._qubits = Qureg([Qubit])
        else:
            raise TypeException("int/list<Qubits/Qureg>/Qureg/Qubit", wires)

        self._idmap = {}
        for idx, qubit in enumerate(self.qubits):
            self._idmap[qubit.id] = idx
        self._gates = []

        self._pointer = -1

    def __del__(self):
        """ release the memory """
        self.gates = None
        self.qubits = None
        self.topology = None
        self.fidelity = None

    # Attributes of the circuit
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
        return CircuitInformation.depth(self.gates)

    def __str__(self):
        circuit_info = {
            "name": self.name,
            "width": self.width(),
            "size": self.size(),
            "depth": self.depth(),
            "1-qubit gates": self.count_1qubit_gate(),
            "2-qubit gates": self.count_2qubit_gate()
        }

        return circuit_info

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
        if method == 'matp':
            from QuICT.tools.drawer import PhotoDrawer
            if filename is None:
                filename = str(self.id) + '.jpg'
            elif '.' not in filename:
                filename += '.jpg'
            photoDrawer = PhotoDrawer()
            photoDrawer.run(self, filename)
        elif method == 'command':
            from QuICT.tools.drawer import TextDrawing
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
        cbits = self.circuit_count_gateType(GateType.measure)
        qasm_string = CircuitInformation.qasm_header(len(self.qubits), cbits)

        cbits = 0
        for gate in self.gates:
            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{gate.targ}] -> c[{cbits}];\n"
                cbits += 1
            else:
                qasm_string += gate.qasm()

        return qasm_string

    def __call__(self, indexes: object):
        """ get a smaller qureg from this circuit

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) list<int>
        Returns:
            Qureg: the qureg correspond to the indexes
        Exceptions:
            IndexDuplicateException: the range of indexes is error.
            TypeException: the type of indexes is error.
        """
        if isinstance(indexes, int):
            assert indexes >= 0 and indexes < self.circuit_width()
        elif isinstance(indexes, list):
            for idx in indexes:
                assert idx >= 0 and idx < self.circuit_width()
        else:
            raise TypeError("only accept int/list[int]")

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

        if isinstance(item, int):
            return self.qubits[item]
        elif isinstance(item, slice):
            qureg_list = self.qubits[item]
            return qureg_list

    # TODO: bug fixed
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
            TypeException: the type of other is wrong
        """
        if not isinstance(targets, Circuit):
            raise TypeError("Only support circuit | circuit.")

        if self.qubits != targets.qubits:
            diff_qubits = self.qubits.diff(targets.qubits)
            for idx, dq in enumerate(diff_qubits):
                targets._idmap[dq.id] = len(targets.qubits) + idx

            targets.qubits = targets.qubits + diff_qubits

        targets.extend(self.gates)

    def append(self, gate):
        """ add a gate to the circuit

        Args:
            gate(BasicGate): the gate to be added to the circuit

        """
        if self._pointer != -1:
            qureg = self.qubits[self._pointer]
            self._pointer = -1
        elif isinstance(gate.assigned_qubits, Qureg):
            qureg = gate.assigned_qubits
        else:
            gate_ctargs = gate.cargs + gate.targs
            if not gate_ctargs:
                if not gate.is_single():
                    raise KeyError(f"{gate.type} need assign qubits to add into circuit.")

                self._add_gate_to_all_qubits(gate)
                return

            qureg = gate_ctargs

        self._add_gate(gate, qureg)

    def _add_gate(self, gate, qureg):
        """ add a gate into some qureg

        Args:
            gate(BasicGate)
            qureg(Qureg/list<Qubit>)
        """
        if isinstance(qureg[0], int):
            qureg = self.qubits(qureg)
            gate.assigned_qubits = qureg
        else:
            gate.cargs = [self._idmap[qureg[idx].id] for idx in range(gate.controls)]
            gate.targs = [self._idmap[qureg[idx].id] for idx in range(gate.controls, gate.controls + gate.targets)]

        gate.update_name(qureg(0).id, len(self.gates))
        self.gates.append(gate)

    def _add_gate_to_all_qubits(self, gate):
        for idx in range(self.circuit_width()):
            new_gate = gate.copy()
            new_gate.targs = [idx]
            new_gate.assigned_qubits = self.qubits(idx)

            self.gates.append(new_gate)

    def extend(self, gates):
        """ add gates to the circuit

        Args:
            gates(list<BasicGate>): the gate to be added to the circuit
        """
        for gate in gates:
            self.append(gate)

    # TODO: optimize
    def sub_circuit(self, targets, start=0, max_size=-1, local=False, remove=False):
        """ get a sub circuit

        Args:
            targets(int/list<int>/tuple<int>/slice): target qubits indexes.
            start(int/qasm_string): the start gate's index, default 0
            max_size(int): max size of the sub circuit, default -1 without limit
            local(bool): whether the slice will stop when meeting an non-commutative gate, default False
            remove(bool): whether deleting the slice gates from origin circuit, default False
        Return:
            Circuit: the sub circuit

        """
        circuit_size = self.circuit_size()
        circuit_width = self.circuit_width()
        if isinstance(targets, slice):
            targets = [i for i in range(circuit_width)][targets]
        targets = list(targets)
        # the mapping from circuit's index to sub-circuit's index
        targets_mapping = [0] * circuit_width
        index = 0
        for target in targets:
            if target < 0 or target >= circuit_width:
                raise Exception('list index out of range')
            targets_mapping[target] = index
            index += 1

        circuit = Circuit(len(targets))
        set_targets = set(targets)
        new_gates = []
        compare_gates = []

        if isinstance(start, str):
            count = 0
            for gate in self.gates:
                if gate.name == start:
                    break
                count += 1
            start = count
        if remove:
            for gate_index in range(start):
                new_gates.append(self.gates[gate_index])
        for gate_index in range(start, circuit_size):
            gate = self.gates[gate_index]
            affectArgs = gate.affectArgs
            set_affectArgs = set(affectArgs)
            # if gate acts on given qubits
            if set_affectArgs <= set_targets:
                compare_gates.append(gate)
                targ = []
                for args in affectArgs:
                    targ.append(targets[args])
                gate | circuit(targ)
                max_size -= 1
            elif len(set_affectArgs.intersection(set_targets)) == 0:
                new_gates.append(gate)
            else:
                if not local:
                    new_gates.append(gate)
                else:
                    commutative = True
                    for goal in compare_gates:
                        if not gate.commutative(goal):
                            commutative = False
                            break
                    if commutative:
                        new_gates.append(gate)
                    else:
                        break
            if max_size == 0:
                if remove:
                    for index in range(gate_index + 1, circuit_size):
                        new_gates.append(self.gates[index])
                break
        if remove:
            for qubit in self.qubits:
                qubit.qState_clear()
            self.set_exec_gates(new_gates)

        return circuit

    def index_for_qubit(self, qubit, ancilla=None) -> int:
        """ find the index of qubit in this circuit

        the index ignored the ancilla qubit

        Args:
            qubit(Qubit): the qubit need to be indexed.
            ancilla(list<Qubit>): the ancillary qubit

        Returns:
            int: the index of the qubit.

        Raises:
            Exception: the qubit is not in the circuit
        """
        if not isinstance(qubit, Qubit):
            raise TypeException("Qubit", now=qubit)
        if ancilla is None:
            ancilla = []
        enterspace = 0
        for i in range(len(self.qubits)):
            if i in ancilla:
                enterspace += 1
            elif self.qubits[i].id == qubit.id:
                return i - enterspace
        raise Exception("the qubit is not in the circuit or it is an ancillary qubit.")

    def random_append(self, rand_size=10, typeList=None):
        """ add some random gate to the circuit

        Args:
            rand_size(int): the number of the gate added to the circuit
            typeList(list<GateType>): the type of gate, default contains CX、ID、Rz、CY、CRz、CH
        """
        self._random_append(rand_size, typeList)

    # TODO: get_n_args, and get_gate fixed
    def _random_append(self, rand_size, typeList):
        if typeList is None:
            typeList = [
                GateType.rx, GateType.ry, GateType.rz,
                GateType.cx, GateType.cy, GateType.crz,
                GateType.ch, GateType.cz, GateType.Rxx,
                GateType.Ryy, GateType.Rzz, GateType.fsim
            ]

        n_qubit = self.width()
        for _ in range(rand_size):
            rand_type = np.random.randint(0, len(typeList))
            gate_type = typeList[rand_type]
            n_pargs, n_targs, n_cargs = get_n_args(gate_type)
            n_affect_args = n_targs + n_cargs
            affect_args = getRandomList(n_affect_args, n_qubit)
            pargs = []
            for _ in range(n_pargs):
                pargs.append(np.random.uniform(0, 2 * np.pi))

            if n_pargs == 0:
                pargs = None

            get_gate(gate_type, affect_args, pargs) | [self[i] for i in affect_args]

    def append_supremacy(self, repeat: int = 1, pattern: str = "ABCDCDAB"):
        """
        Add a supremacy circuit to the circuit

        Args:
            repeat(int): the number of two-qubit gates' sequence
            pattern(str): indicate the two-qubit gates' sequence
        """
        self._supremacy_append(repeat, pattern)

    # TODO: bug fixed
    def _supremacy_append(self, repeat, pattern):
        qubits = len(self.qubits)
        supremacy_layout = SupremacyLayout(qubits)

        self._add_gate_to_all_qubits(H)

        for i in range(repeat * len(pattern)):
            for q in range(qubits):
                gate_type = SUPREMACY_GATE_SET[np.random.randint(0, 3)]

                get_gate(gate_type) | self.circuit(q)

            current_pattern = pattern[i % (len(pattern))]
            if current_pattern not in "ABCD":
                raise KeyError(f"Unsupported pattern {pattern[i]}, please use one of 'A', 'B', 'C', 'D'.")

            edges = supremacy_layout.get_edges_by_pattern(current_pattern)
            for e in edges:
                FSim([np.pi / 2, np.pi / 6]) | self([int(e[0]), int(e[1])])

        self._add_gate_to_all_qubits(Measure)

    def matrix_product_to_circuit(self, gate) -> np.ndarray:
        """ extend a gate's matrix in the all circuit unitary linear space

        gate's matrix tensor products some identity matrix.

        Args:
            gate(BasicGate): the gate to be extended.

        """
        return self._inner_matrix_product_to_circuit(self, gate)

    def _inner_matrix_product_to_circuit(self, gate) -> np.ndarray:
        q_len = len(self.qubits)
        n = 1 << len(self.qubits)

        new_values = np.zeros((n, n), dtype=np.complex128)
        targs = gate.targs
        cargs = gate.cargs
        if not isinstance(targs, list):
            targs = [targs]
        if not isinstance(cargs, list):
            cargs = [cargs]
        targs = np.append(
            np.array(cargs, dtype=int).ravel(),
            np.array(targs, dtype=int).ravel()
        )
        targs = targs.tolist()
        xor = (1 << q_len) - 1
        if not isinstance(targs, list):
            raise Exception("unknown error")
        matrix = gate.compute_matrix.reshape(1 << len(targs), 1 << len(targs))
        datas = np.zeros(n, dtype=int)
        for i in range(n):
            nowi = 0
            for kk in range(len(targs)):
                k = q_len - 1 - targs[kk]
                if (1 << k) & i != 0:
                    nowi += (1 << (len(targs) - 1 - kk))
            datas[i] = nowi
        for i in targs:
            xor = xor ^ (1 << (q_len - 1 - i))
        for i in range(n):
            nowi = datas[i]
            for j in range(n):
                nowj = datas[j]
                if (i & xor) != (j & xor):
                    continue
                new_values[i][j] = matrix[nowi][nowj]

        return new_values
