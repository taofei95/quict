#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py

from typing import List
import numpy as np
from enum import Enum

from .gate_type import GateType
from .circuit_matrix import CircuitMatrix, get_gates_order_by_depth


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

    def get_gates_order_by_depth(self) -> List[List]:
        """ Order the gates of circuit by its depth layer

        Returns:
            List[List[BasicGate]]: The list of gates which at same layers in circuit.
        """
        return get_gates_order_by_depth(self.gates)

    def matrix(self, device: str = "CPU", mini_arg: int = 0) -> np.ndarray:
        """ Generate the circuit's unitary matrix which compose by all quantum gates' matrix in current circuit.

        Args:
            device (str, optional): The device type for generate circuit's matrix, one of [CPU, GPU]. Defaults to "CPU".
            mini_arg (int, optional): The minimal qubit args, only use for CompositeGate local mode. Default to 0.
        """
        assert device in ["CPU", "GPU"]
        circuit_matrix = CircuitMatrix(device)

        if not self._gates:
            return None

        return circuit_matrix.get_unitary_matrix(self.gates, self.width(), mini_arg)

    def convert_precision(self):
        """ Convert all gates in Cicuit/CompositeGate into single precision. """
        for gate in self.gates:
            gate.convert_precision()


class CircuitMode(Enum):
    Clifford = "Clifford"
