#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py
from enum import Enum
from typing import List
import numpy as np

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
        combined_gates = [gate & targs for gate, targs, _ in self._gates]

        return combined_gates

    def __init__(self, name: str):
        self._name = name
        self._gates = []
        self._pointer = None
        self._precision = np.complex128

    def size(self) -> int:
        """ the number of gates in the circuit/CompositeGate

        Returns:
            int: the number of gates in circuit
        """
        tsize = 0
        for _, _, size in self._gates:
            tsize += size

        return tsize

    def width(self):
        """ the number of qubits in circuit

        Returns:
            int: the number of qubits in circuit
        """
        return len(self._qubits)

    def depth(self) -> int:
        """ the depth of the circuit/CompositeGate.

        Returns:
            int: the depth
        """
        depth = np.zeros(self.width(), dtype=int)
        for gate, targs, gsize in self._gates:
            gdepth = 1 if gsize == 1 else gate.depth()
            depth[targs] = np.max(depth[targs]) + gdepth

        return np.max(depth)

    def count_2qubit_gate(self) -> int:
        """ the number of the two qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the two qubit gates
        """
        count = 0
        for gate, _, _ in self._gates:
            if gate.controls + gate.targets == 2:
                count += 1

        return count

    def count_1qubit_gate(self) -> int:
        """ the number of the one qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the one qubit gates
        """
        count = 0
        for gate, _, _ in self._gates:
            if gate.controls + gate.targets == 1:
                count += 1

        return count

    def count_gate_by_gatetype(self, gate_type: GateType) -> int:
        """ the number of the gates which are some type in the circuit/CompositeGate

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type
        """
        count = 0
        for gate, _, size in self._gates:
            if size > 1:
                count += gate.count_gate_by_gatetype(gate_type)
            elif gate.type == gate_type:
                count += 1

        return count

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

    def qasm(self, output_file: str = None, header_required: bool = True):
        """ The qasm of current CompositeGate/Circuit.

        Args:
            output_file (str): The output qasm file's name or path

        Returns:
            str: The string of Circuit/CompositeGate's qasm.
        """
        qasm_string = ""
        qreg = self.width()
        creg = min(self.count_gate_by_gatetype(GateType.measure), qreg)
        if creg == 0:
            creg = qreg

        if header_required:
            qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            qasm_string += f"qreg q[{qreg}];\n"
            qasm_string += f"creg c[{creg}];\n"

        cbits = 0
        for gate, targs, size in self._gates:
            if size > 1:
                qasm_string += gate.qasm(header_required=False)
            else:
                if gate.qasm_name == "measure":
                    qasm_string += f"measure q[{targs}] -> c[{cbits}];\n"
                    cbits += 1
                    cbits = cbits % creg
                else:
                    qasm_string += gate.qasm(targs)

        if output_file is not None:
            with open(output_file, 'w+') as of:
                of.write(qasm_string)

        return qasm_string

    # TODO: refactoring, may remove
    def get_lastcall_for_each_qubits(self) -> List[GateType]:
        lastcall_per_qubits = [None] * self.width()
        inside_qargs = []
        for i in range(self.size() - 1, -1, -1):
            gate_args = self._gates[i][1]
            gate_type = self._gates[i][0].type
            for garg in gate_args:
                if lastcall_per_qubits[garg] is None:
                    lastcall_per_qubits[garg] = gate_type
                    inside_qargs.append(garg)

            if len(inside_qargs) == self.width():
                break

        return lastcall_per_qubits

    # TODO: refactoring
    def convert_precision(self):
        """ Convert all gates in Cicuit/CompositeGate into single precision. """
        for gate in self.gates:
            if hasattr(gate, "convert_precision"):
                gate.convert_precision()

        self._precision = np.complex64 if self._precision == np.complex128 else np.complex128

    def gate_decomposition(self) -> list:
        decomp_gates = []
        for gate, qidxes, size in self._gates:
            if size > 1:
                decomp_gates += gate.gate_decomposition()
                continue

            if hasattr(gate, "build_gate"):
                cgate = gate.build_gate()
                if cgate is not None:
                    cgate & qidxes
                    decomp_gates += cgate._gates
                    continue

            decomp_gates.append((gate, qidxes, size))

        self._gates = decomp_gates
        return self._gates


class CircuitMode(Enum):
    Clifford = "Clifford"
    CliffordRz = "CliffordRz"
    Arithmetic = 'Arithmetic'
    Misc = "Misc"
