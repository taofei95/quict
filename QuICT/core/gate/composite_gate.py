#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 9:36 下午
# @Author  : Han Yu, Li Kaiqi
# @File    : composite_gate.py
import numpy as np

from QuICT.core.qubit import Qureg, Qubit
from QuICT.core.gate.gate import BasicGate
from QuICT.core.utils import GateType, CircuitInformation, matrix_product_to_circuit, CGATE_LIST


# global composite gate id count
cgate_id = 0


class CompositeGate:
    """ Implement a group of gate

    Attributes:
        gates (list<BasicGate>): gates within this composite gate
    """
    @property
    def gates(self):
        return self._gates

    @property
    def name(self):
        return self._name

    def __enter__(self):
        global CGATE_LIST
        CGATE_LIST.append(self)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global CGATE_LIST
        CGATE_LIST.remove(self)

        return True

    def __init__(self, name: str = None, gates: list = None):
        """ initial a CompositeGate with gate(s)

        Args:
            name [str]: the name of composite gate
            gates List[BasicGate]: The gates are added into this composite gate
        """
        global cgate_id
        self._id = cgate_id
        cgate_id += 1

        self._name = name if name else f"composite_gate_{self._id}"
        self._gates = []
        self._min_qubit = np.inf
        self._max_qubit = 0
        self._pointer = -1

        if gates:
            self.extend(gates)

    def __and__(self, targets):
        """ assign qubits or indexes for given gates

        Args:
            targets ([int/qubit/list[int]/qureg]): qubit describe
        """
        if isinstance(targets, int):
            targets = [targets]

        if isinstance(targets, Qubit):
            targets = Qureg(targets)

        if len(targets) != self._max_qubit:
            raise ValueError("The number of assigned qubits or indexes must be equal to gate's width.")

        self._mapping(targets)
        if CGATE_LIST:
            CGATE_LIST[-1].extend(self.gates)

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
            TypeException: the type of other is wrong
        """
        try:
            targets.extend(self.gates)
        except Exception as e:
            raise TypeError(f"Only support circuit and composite gate. {e}")

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
            TypeException: the type of other is wrong
        """
        self.inverse()
        try:
            targets.extend(self.gates)
        except Exception as e:
            raise TypeError(f"Only support circuit for gateSet ^ circuit. {e}")

    def __getitem__(self, item):
        """ get gates from this composite gate

        Args:
            item(int/slice): slice passed in.

        Return:
            [BasicGates]: the gates
        """
        return self._gates[item]

    def __call__(self, indexes: list):
        if isinstance(indexes, int):
            indexes = [indexes]

        self._update_qubit_limit(indexes)
        self._pointer = indexes
        return self

    def _mapping(self, targets: Qureg):
        """ remapping the gates' affectArgs

        Args:
            targets(Qureg/List): the related qubits
        """
        for gate in self._gates:
            args_index = gate.cargs + gate.targs
            if isinstance(targets, Qureg):
                target_qureg = targets(args_index)
                gate.assigned_qubits = target_qureg
                gate.update_name(target_qureg[0].id)
            else:
                gate.cargs = [targets[carg] for carg in gate.cargs]
                gate.targs = [targets[targ] for targ in gate.targs]

    def _update_qubit_limit(self, indexes: list):
        for idx in indexes:
            assert idx >= 0 and isinstance(idx, int)
            if idx >= self._max_qubit:
                self._max_qubit = idx + 1

            if idx < self._min_qubit:
                self._min_qubit = idx

    def extend(self, gates: list):
        for gate in gates:
            self.append(gate, is_extend=True)

        self._pointer = -1

    def append(self, gate, is_extend: bool = False, insert_idx: int = -1):
        gate = gate.copy()

        if self._pointer != -1:
            qubit_index = self._pointer[:]
            gate_args = gate.controls + gate.targets
            if len(self._pointer) > gate_args:
                gate.cargs = [qubit_index[carg] for carg in gate.cargs]
                gate.targs = [qubit_index[targ] for targ in gate.targs]
            elif len(self._pointer) == gate_args:
                gate.cargs = qubit_index[:gate.controls]
                gate.targs = qubit_index[gate.controls:]
            else:
                raise KeyError(f"{gate.type} need {gate_args} indexes, but given {len(self._pointer)}")

            if not is_extend:
                self._pointer = -1
        else:
            qubit_index = gate.cargs + gate.targs
            if not qubit_index:
                raise KeyError(f"{gate.type} need qubit indexes to add into Composite Gate.")

            self._update_qubit_limit(qubit_index)

        if insert_idx == -1:
            self._gates.append(gate)
        else:
            self._gates.insert(insert_idx, gate)

    def left_append(self, gate):
        self.append(gate, insert_idx=0)

    def left_extend(self, gates: list):
        for idx, gate in enumerate(gates):
            self.append(gate, is_extend=True, insert_idx=idx)

        self._pointer = -1

    def width(self):
        """ the number of qubits applied by gates

        Returns:
            int: the number of qubits applied by gates
        """
        return self._max_qubit

    def size(self):
        """ the size of the gates

        Returns:
            int: the number of gates in gates
        """
        return len(self._gates)

    def count_2qubit_gate(self):
        """ the number of the two qubit gates in the set

        Returns:
            int: the number of the two qubit gates in the set
        """
        return CircuitInformation.count_2qubit_gate(self.gates)

    def count_1qubit_gate(self):
        """ the number of the one qubit gates in the set

        Returns:
            int: the number of the one qubit gates in the set
        """
        return CircuitInformation.count_1qubit_gate(self.gates)

    def count_gate_by_gatetype(self, gate_type):
        """ the number of the gates which are some type in the set

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type in the circuit
        """
        return CircuitInformation.count_gate_by_gatetype(self.gates, gate_type)

    def depth(self):
        """ the depth of the circuit for some gate.

        Args:
            gateTypes(list<GateType>):
                the types to be count into depth calculate
                if count all type of gates, leave it being None.

        Returns:
            int: the depth of the circuit
        """
        return CircuitInformation.depth(self.gates)

    def __str__(self):
        cgate_info = {
            "width": self.width(),
            "size": self.size(),
            "depth": self.depth(),
            "1-qubit gates": self.count_1qubit_gate(),
            "2-qubit gates": self.count_2qubit_gate(),
            "gates detail": []
        }

        for gate in self.gates:
            cgate_info["gates detail"].append(str(gate))

        return str(cgate_info)

    def qasm(self):
        """ get OpenQASM 2.0 describe for the composite gate

        Returns:
            str: OpenQASM 2.0 describe
        """
        qreg = self.width()
        creg = self.count_gate_by_gatetype(GateType.measure)

        return CircuitInformation.qasm(qreg, creg, self.gates)

    def inverse(self):
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        inverse_cgate = CompositeGate()
        inverse_gates = [gate.inverse() for gate in self._gates[::-1]]
        inverse_cgate.extend(inverse_gates)

        return inverse_cgate

    def matrix(self, local: bool = False):
        """ matrix of these gates

        Args:
            local: whether regards the min_qubit as the 0's qubit

        Returns:
            np.ndarray: the matrix of the gates
        """
        if not self._gates:
            return None

        if local and isinstance(self._min_qubit, int):
            min_value = self._min_qubit
        else:
            min_value = 0

        matrix = np.eye(1 << (self._max_qubit - min_value))
        for gate in self.gates:
            if gate.is_special() and gate.type != GateType.unitary:
                raise TypeError(f"Cannot combined the gate matrix with special gate {gate.type}")

            matrix = np.matmul(matrix_product_to_circuit(
                gate.matrix, gate.cargs + gate.targs, self._max_qubit, min_value
            ), matrix)

        return matrix

    def equal(self, target, ignore_phase=True, eps=1e-7):
        """ whether is equally with target or not.

        Args:
            target(gateSet/BasicGate/Circuit): the target
            ignore_phase(bool): ignore the global phase
            eps(float): the tolerable error

        Returns:
            bool: whether the gateSet is equal with the targets
        """
        self_matrix = self.matrix()
        if isinstance(target, CompositeGate):
            target_matrix = target.matrix()
        elif isinstance(target, BasicGate):
            target_matrix = target.matrix
        else:
            temp_cg = CompositeGate()
            for gate in target.gates:
                gate | temp_cg

            target_matrix = temp_cg.matrix()

        if ignore_phase:
            shape = self_matrix.shape
            rotate = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if abs(self_matrix[i, j]) > eps:
                        rotate = target_matrix[i, j] / self_matrix[i, j]
                        break

                if rotate != 0:
                    break

            if rotate == 0 or abs(abs(rotate) - 1) > eps:
                return False

            self_matrix = self_matrix * np.full(shape, rotate)

        return np.allclose(self_matrix, target_matrix, rtol=eps, atol=eps)
