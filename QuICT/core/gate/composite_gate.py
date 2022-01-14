#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 9:36 下午
# @Author  : Han Yu
# @File    : composite_gate
import numpy as np

from QuICT.core.qubit import Qureg, Qubit
from QuICT.core.utils import GateType, CircuitInformation, matrix_product_to_circuit


# global composite gate id count
cgate_id = 0
CGATE_LIST = []


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

    def __init__(self, wires: int, name: str = None, gates: list = None):
        """ initial a CompositeGate with gate(s)

        Args:
            qubits [BasicGate]: the qubits which make up the qureg, it can have below form,
        """
        global cgate_id
        self._id = cgate_id
        cgate_id += 1

        self._name = name if name else f"composite_gate_{self._id}"
        self._qubits = wires
        self._gates = []
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

        assert len(targets) == self._qubits
        self._mapping(targets)

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

        for idx in indexes:
            assert idx >= 0 and idx < self._qubits

        self._pointer = indexes
        return self

    def extend(self, gates: list):
        for gate in gates:
            self.append(gate, is_extend=True)

        self._pointer = -1

    def append(self, gate, is_extend: bool = False):
        gate = gate.copy()

        if self._pointer != -1:
            qubit_index = list(self._pointer)
            if len(qubit_index) > gate.controls + gate.targets:
                gate.cargs = [qubit_index[carg] for carg in gate.cargs]
                gate.targs = [qubit_index[targ] for targ in gate.targs]
            else:
                gate.cargs = qubit_index[:gate.controls]
                gate.targs = qubit_index[gate.controls:]

            if not is_extend:
                self._pointer = -1
        else:
            qubit_index = gate.cargs + gate.targs
            if not qubit_index:
                raise KeyError(f"{gate.type} need qubit indexes to add into Composite Gate.")

            for idx in qubit_index:
                assert idx >= 0 and idx < self._qubits

        self._gates.append(gate)

    def width(self):
        """ the number of qubits applied by gates

        Returns:
            int: the number of qubits applied by gates
        """
        return self._qubits

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
            "gates detail": {}
        }

        for gate in self.gates:
            cgate_info["gates detail"][gate.name] = str(gate)

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
        self._gates = [gate.inverse() for gate in self._gates]

    def matrix(self):
        """ matrix of these gates

        Args:
            local: whether regards the min_qubit as the 0's qubit

        Returns:
            np.ndarray: the matrix of the gates
        """
        matrix = np.eye(1 << self._qubits)
        for gate in self.gates:
            if gate.is_special():
                raise TypeError(f"Cannot combined the gate matrix with special gate {gate.type}")

            matrix = np.matmul(matrix_product_to_circuit(self._qubits, gate), matrix)

        return matrix

    def equal(self, target, ignore_phase=True, eps=1e-7):   # TODO: refactoring
        """ whether is equally with target or not.

        Args:
            target(gateSet/BasicGate/Circuit): the target
            ignore_phase(bool): ignore the global phase
            eps(float): the tolerable error

        Returns:
            bool: whether the gateSet is equal with the targets
        """
        target = CompositeGate(target, with_copy=True)
        self_matrix = self.matrix()
        target_matrix = target.matrix()
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
