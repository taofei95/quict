#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 10:31
# @Author  : Han Yu
# @File    : _circuit_computing.py

import random


def getRandomList(count, upper_bound):
    """ get `count` number from 0, 1, ..., `upper_bound - 1` randomly.

    Args:
        count(int)
        upper_bound(int)
    Returns:
        list<int>: the list of l random numbers
    """
    _rand = [i for i in range(upper_bound)]
    for i in range(upper_bound - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:count]


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
    def depth(gates):
        layers = []
        for gate in gates:
            now = set(gate.cargs) | set(gate.targs)
            for i in range(len(layers) - 1, -2, -1):
                if i == -1 or len(now & layers[i]) > 0:
                    if i + 1 == len(layers):
                        layers.append(set())
                    layers[i + 1] |= now
        return len(layers)

    @staticmethod
    def qasm_header(qreg, creg):
        header = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        header += f"qreg q[{qreg}];\n"
        header += f"creg c[{creg}];\n"

        return header
