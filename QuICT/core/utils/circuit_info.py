#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py

import numpy as np


class CircuitInformation:
    @staticmethod
    def count_2qubit_gate(gates):
        count = 0
        for gate in gates:
            if gate.controls + gate.targets == 2:
                count += 1
        return count

    @staticmethod
    def count_1qubit_gate(gates):
        count = 0
        for gate in gates:
            if gate.is_single():
                count += 1
        return count

    @staticmethod
    def count_gate_by_gatetype(gates, gate_type):
        count = 0
        for gate in gates:
            if gate.type == gate_type:
                count += 1
        return count

    @staticmethod
    def depth(gates, width):
        depth = np.zeros(width, dtype=int)
        for gate in gates:
            targs = gate.cargs + gate.targs
            depth[targs] = np.max(depth[targs]) + 1
        return np.max(depth)

    @staticmethod
    def qasm(qreg, creg, gates):
        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_string += f"qreg q[{qreg}];\n"
        qasm_string += f"creg c[{creg}];\n"

        cbits = 0
        for gate in gates:
            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{gate.targ}] -> c[{cbits}];\n"
                cbits += 1
            else:
                qasm_string += gate.qasm()

        return qasm_string
