#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 9:36 下午
# @Author  : Han Yu
# @File    : composite_gate
from QuICT.core.circuit import Circuit, CircuitInformation
from QuICT.core.utils import GateType

from .gate import *
from ..qubit import Qureg


class CompositeGate:
    """ Implement a group of gate

    Attributes:
        Qureg (Qureg): the related qubits
        gates (list<BasicGate>): gates within this composite gate
    """
    @property
    def gates(self):
        return self.gates

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    # TODO: allow no qureg
    def __init__(self, gates: list = None, with_copy: bool = True):
        """ initial a CompositeGate with gate(s)

        Args:
            qubits [BasicGate]: the qubits which make up the qureg, it can have below form,
        """
        self._gates = gates if gates else []
        self._is_copy = with_copy

    def width(self):
        """ the number of qubits applied by gates

        Returns:
            int: the number of qubits applied by gates
        """
        return len(self._qubits)

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
    
    def gate_info(self):
        cgate_info = {
            "width": self.width(),
            "size": self.size(),
            "depth": self.depth(),
            "1-qubit gates": self.count_1qubit_gate(),
            "2-qubit gates": self.count_2qubit_gate(),
            "gates detail": {} 
        }

        for gate in self.gates:
            cgate_info["gates detail"][gate.name] = gate.gate_info

        return cgate_info

    def qasm(self):
        """ get OpenQASM 2.0 describe for the composite gate

        Returns:
            str: OpenQASM 2.0 describe
        """
        qreg = self.width(),
        creg = self.count_gate_by_gatetype(GateType.measure)

        return CircuitInformation.qasm(qreg, creg, self.gates)

    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "gateSet | circuit"
        to add the gate of gateSet into the circuit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
        Raise:
            TypeException: the type of other is wrong
        """
        assert isinstance(targets, Circuit)

        targets.extend(self.gates)

    def __xor__(self, targets):
        """deal the operator '^'

        Use the syntax "gateSet ^ circuit"
        to add the gate of gateSet's inverse into the circuit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
        Raise:
            TypeException: the type of other is wrong
        """
        assert isinstance(targets, Circuit)

        self.inverse()
        targets.extend(self.gates)

    # TODO: return gate or qubit ?
    def __getitem__(self, item):
        """ to fit the slice operator, overloaded this function.

        get a smaller qureg/qubit from this qureg

        Args:
            item(int/slice): slice passed in.
        Return:
            Qubit/Qureg: the result or slice
        """
        return self._gates[item]

    def inverse(self):
        """ the inverse of CompositeGate

        Returns:
            CompositeGate: the inverse of the gateSet
        """
        for gate in self._gates:
            gate.inverse()

    # TODO: refactoring later
    def matrix(self, local=False):
        """ matrix of these gates

        Args:
            local: whether regards the min_qubit as the 0's qubit

        Returns:
            np.ndarray: the matrix of the gates
        """
        min_qubit = -1
        max_qubit = -1
        for gate in self.gates:
            for arg in gate.affectArgs:
                if min_qubit == -1:
                    min_qubit = arg
                else:
                    min_qubit = min(min_qubit, arg)
                if max_qubit == -1:
                    max_qubit = arg
                else:
                    max_qubit = max(max_qubit, arg)

        if min_qubit == -1:
            return np.eye(2, dtype=np.complex128)

        if local:
            q_len = max_qubit - min_qubit + 1
        else:
            q_len = max_qubit + 1

        n = 1 << q_len
        result = np.eye(n, dtype=np.complex128)
        for gate in self.gates:
            new_values = np.zeros((n, n), dtype=np.complex128)
            targs = gate.affectArgs
            for i in range(len(targs)):
                targs[i] -= min_qubit

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

                    new_values[i][j] = matrix[nowi, nowj]

            result = np.dot(new_values, result)

        return result

    # TODO: refactoring later
    def equal(self, target, ignore_phase=True, eps=1e-7):
        """

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

    def remapping(self, mapping):
        """ remapping the gates' affectArgs

        Args:
            mapping(list): the mapping function

        Returns:

        """
        for gate in self._gates:
            affectArgs = []
            for arg in gate.assigned_qubits:
                affectArgs.append(mapping[arg])

            gate.assigned_qubits = affectArgs
