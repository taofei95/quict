#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py

import numpy as np
from enum import Enum

from .gate_type import GateType


class CircuitBased(object):
    """ Based Class for Circuit and Composite Gate. """
    @property
    def name(self) -> int:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def gates(self) -> list:
        return self._gates

    @gates.setter
    def gates(self, gates):
        self._gates = gates

    def __init__(self, name: str):
        self._name = name
        self._gates = []
        self._gate_type = {}        # gate_type: # of gates
        self._pointer = None

    def size(self) -> int:
        """ the number of gates in the circuit/CompositeGate

        Returns:
            int: the number of gates in circuit
        """
        return len(self._gates)

    def width(self):
        """ the number of qubits in circuit

        Returns:
            int: the number of qubits in circuit
        """
        return len(self.qubits)

    def depth(self) -> int:
        """ the depth of the circuit/CompositeGate.

        Returns:
            int: the depth
        """
        depth = np.zeros(self.width(), dtype=int)
        for gate in self._gates:
            targs = gate.cargs + gate.targs
            depth[targs] = np.max(depth[targs]) + 1

        return np.max(depth)

    def count_2qubit_gate(self) -> int:
        """ the number of the two qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the two qubit gates
        """
        count = 0
        for gate in self._gates:
            if gate.controls + gate.targets == 2:
                count += 1

        return count

    def count_1qubit_gate(self) -> int:
        """ the number of the one qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the one qubit gates
        """
        count = 0
        for gate in self._gates:
            if gate.is_single():
                count += 1

        return count

    def count_gate_by_gatetype(self, gate_type: GateType) -> int:
        """ the number of the gates which are some type in the circuit/CompositeGate

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type
        """
        if gate_type in self._gate_type.keys():
            return self._gate_type[gate_type]

        return 0

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

    def qasm(self):
        qreg = self.width()
        creg = self.count_gate_by_gatetype(GateType.measure)
        if creg == 0:
            creg = qreg

        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_string += f"qreg q[{qreg}];\n"
        qasm_string += f"creg c[{creg}];\n"

        cbits = 0
        for gate in self._gates:
            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{gate.targ}] -> c[{cbits}];\n"
                cbits += 1
            else:
                qasm_string += gate.qasm()

        return qasm_string


class CircuitMode(Enum):
    Clifford = "Clifford"
