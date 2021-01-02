#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/1 8:11 下午
# @Author  : Han Yu
# @File    : cnot_local_force.py

import numpy as np

from .._optimization import Optimization
from QuICT.core import *
from QuICT.qcda.optimization import CnotForceBfs

def traver_with_fix_qubits(gates: list, fix: set):
    """ local optimize for fix qubits
    Args:
        gates(list<CXGate>): the gates to be optimized
        fix(set<int>): the fix qubits
    Returns:
        list<CXGate>: results after optimization
    """

    mapping = {}
    back_map = {}
    mapping_id = 0
    for element in fix:
        mapping[element] = mapping_id
        back_map[mapping_id] = element
        mapping_id += 1

    output = []
    local_list = []
    for gate in gates:
        fix_in = int(gate.carg in fix) + int(gate.targ in fix)
        if fix_in == 2:
            local_list.append(gate)
        elif fix_in == 0:
            output.append(gate)
        else:
            if len(local_list) != 0:
                circuit = Circuit(len(fix))
                for local_gate in local_list:
                    CX | circuit([mapping[local_gate.carg], mapping[local_gate.targ]])
                new_gates = CnotForceBfs.run(circuit)
                for local_gate in new_gates:
                    new_gate = CX.copy()
                    new_gate.cargs = [back_map[local_gate.carg], back_map[local_gate.targ]]
                    output.append(new_gate)
            output.append(gate)
    return output

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
