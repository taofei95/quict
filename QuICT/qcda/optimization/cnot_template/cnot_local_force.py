#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/1 8:11 下午
# @Author  : Han Yu
# @File    : cnot_local_force.py

import numpy as np

from .._optimization import Optimization
from QuICT.core import *
from QuICT.qcda.optimization import CnotForceBfs

def local_optimize(gates: list, index_set: set):
    return []

def add_set(index_set: set, gate: BasicGate, max_local_qubits):
    """ add gate into local set

    :param index_set:
    :param gate:
    :param max_local_qubits:
    :return:
    """
    if gate.type() != GATE_ID["CX"]:
        raise Exception("input gates should contain only CX gates")
    if gate.carg not in index_set:
        if len(index_set) >= max_local_qubits:
            return False
        index_set.add(gate.carg)
    if gate.targ not in index_set:
        if len(index_set) >= max_local_qubits:
            return False
        index_set.add(gate.targ)
    return True

def traver(input: list, max_local_qubits = 5):
    """ find the best circuit by bfs

    Args:
        input(list<BasicGate>): input circuit
        max_local_qubits(int): the max number of qubits

    Returns:
        list<BasicGate>: gates after optimization
        bool: whether the gates decrease

    """
    output = []
    renew = False
    last = 0
    index_set = set()
    length_input = len(input)
    for i in range(length_input):
        succ = add_set(index_set, input[i], max_local_qubits)
        if not succ:
            new_gates, succ = local_optimize(input[last:i], index_set)
            last = i
            if not succ:
                output.extend(input[last:i])
            else:
                output.extend(new_gates)
                renew = True
    if last != length_input:
        new_gates, succ = local_optimize(input[last:], index_set)
        if not succ:
            output.extend(input[last:])
        else:
            output.extend(new_gates)
            renew = True
    return output, renew

def solve(input: list):
    """ optimize the circuit locally

    Args:
        input(list<BasicGate>): input circuit

    Returns:
        Circuit: optimal circuit

    """
    gates = input.gates
    while True:
        gates, renew = traver(gates)
        if not renew:
            break
    return gates

class CnotLocalForceBfs(Optimization):
    """ use bfs to optimize the cnot circuit

    """
    @staticmethod
    def _run(circuit : Circuit, *pargs):
        """
        circuit(Circuit): the circuit to be optimize
        *pargs: other parameters
        """
        gates = circuit.gates
        gates = solve(gates)
        circuit.clear()
        circuit.extend(gates)
