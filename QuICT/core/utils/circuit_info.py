#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py
from enum import Enum
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
    def precision(self) -> str:
        return self._precision

    @property
    def gates(self) -> list:
        """ Return the list of BasicGate/CompositeGate/Operator in the current circuit. \n
        *Warning*: this is slowly due to the copy of gates, you can use self.fast_gates to
        get list of tuple(gate, qidxes, size) for further using. 
        """
        combined_gates = [gate.copy() & targs for gate, targs, _ in self._gates]

        return combined_gates

    @property
    def fast_gates(self) -> list:
        """ Return the list of tuple(gates' info) in the current circuit. it contains the gate, 
        the qubit indexes and the gate's size."""
        return self._gates

    def flatten_gates(self, decomposition: bool = False) -> list:
        """ Return the list of BasicGate/Operator. """
        flatten_gates = []
        for gate, qidxes, size in self._gates:
            gate = gate.copy() & qidxes
            if size > 1:
                flatten_gates.extend(gate.flatten_gates(decomposition))
            else:
                if decomposition:
                    cgate = gate.build_gate()
                    if cgate is not None:
                        flatten_gates.extend(cgate.gates)
                        continue

                flatten_gates.append(gate.copy() & qidxes)

        return flatten_gates

    def __init__(self, name: str):
        self._name = name
        self._gates = []
        self._pointer = None
        self._precision = "double"

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

    def depth(self, depth_per_qubits: bool = False) -> int:
        """ the depth of the circuit.

        Returns:
            int: the depth
        """
        depth = np.zeros(self.width(), dtype=int)
        for gate, targs, gsize in self._gates:
            if gsize > 1:
                gdepth = gate.depth(True)
                for i, targ in enumerate(targs):
                    depth[targ] += gdepth[i]
            else:
                depth[targs] = np.max(depth[targs]) + 1

        return np.max(depth) if not depth_per_qubits else depth

    def count_2qubit_gate(self) -> int:
        """ the number of the two qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the two qubit gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size > 1:
                count += gate.count_2qubit_gate()
                continue

            if gate.controls + gate.targets == 2:
                count += 1

        return count

    def count_1qubit_gate(self) -> int:
        """ the number of the one qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the one qubit gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size > 1:
                count += gate.count_1qubit_gate()
                continue

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

    def qasm(self, output_file: str = None):
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

        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_string += f"qreg q[{qreg}];\n"
        qasm_string += f"creg c[{creg}];\n"

        cbits = 0
        for gate, targs, size in self._gates:
            if size > 1:
                qasm_string += gate.qasm_gates_only(creg, cbits, targs)
                continue

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

    def set_precision(self, precision: str):
        """ Set precision for Cicuit/CompositeGate

        Args:
            precision(str): The precision of Circuit/CompositeGate, should be one of [single, double]
        """
        assert precision in ["single", "double"], "Circuit's precision should be one of [double, single]"
        self._precision = precision

    def gate_decomposition(self, self_flatten: bool = True) -> list:
        decomp_gates = []
        for gate, qidxes, size in self._gates:
            if size > 1:
                decomp_gates += gate.gate_decomposition()
                continue

            cgate = gate.build_gate()
            if cgate is not None:
                cgate & qidxes
                decomp_gates += cgate._gates
                continue

            decomp_gates.append((gate, qidxes, size))

        if not self_flatten:
            return decomp_gates
        else:
            self._gates = decomp_gates
            return self._gates


class CircuitMode(Enum):
    Clifford = "Clifford"
    CliffordRz = "CliffordRz"
    Arithmetic = 'Arithmetic'
    Misc = "Misc"
