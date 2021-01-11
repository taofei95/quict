#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/1 8:11 下午
# @Author  : Han Yu
# @File    : cnot_local_force.py

from itertools import combinations

import numpy as np

from .._optimization import Optimization
from .cnot_force import CnotForceBfs
from .cnot_store_force import CnotStoreForceBfs
from QuICT.core import *


def traver_with_fix_qubits(gates: list, fix: set, store):
    """ local optimize for fix qubits
    Args:
        gates(list<CXGate>): the gates to be optimized
        fix(set<int>): the fix qubits
        store(bool): whether work with local data
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
                print((CnotStoreForceBfs if store else CnotForceBfs))
                new_circuit = (CnotStoreForceBfs if store else CnotForceBfs).run(circuit)
                for local_gate in new_circuit.gates:
                    new_gate = CX.copy()
                    new_gate.cargs = [back_map[local_gate.carg]]
                    new_gate.targs = [back_map[local_gate.targ]]
                    output.append(new_gate)
                local_list = []
            output.append(gate)

    if len(local_list) != 0:
        circuit = Circuit(len(fix))
        for local_gate in local_list:
            CX | circuit([mapping[local_gate.carg], mapping[local_gate.targ]])
        print((CnotStoreForceBfs if store else CnotForceBfs))
        new_circuit = (CnotStoreForceBfs if store else CnotForceBfs).run(circuit)
        for local_gate in new_circuit.gates:
            new_gate = CX.copy()
            new_gate.cargs = [back_map[local_gate.carg]]
            new_gate.targs = [back_map[local_gate.targ]]
            output.append(new_gate)

    return output

def traver(input: list, width, store):
    """ find the best circuit by bfs

    Args:
        input(list<BasicGate>): input circuit
        width(int): circuit_width
        store(bool): whether work with local data
        max_local_qubits(int): the max number of qubits

    Returns:
        list<BasicGate>: gates after optimization

    """
    all_list = [i for i in range(width)]
    max_local_qubits = 5 if store else 4
    for comb in combinations(all_list, min(width, max_local_qubits)):
        input = traver_with_fix_qubits(input, set(comb), store)
    return input

def solve(gates: list, width, store):
    """ optimize the circuit locally

    Args:
        gates(list<BasicGate>): input circuit
        width(int): circuit_width
        store(bool): whether work with local data

    Returns:
        Circuit: optimal circuit

    """
    last_length = len(gates)
    while True:
        gates = traver(gates, width, store)
        new_length = len(gates)
        if last_length == new_length:
            break
        last_length = new_length
    circuit = Circuit(width)
    circuit.extend(gates)
    return gates

class CnotLocalForceBfs(Optimization):
    """ use bfs to optimize the cnot circuit

    """
    @staticmethod
    def _run(circuit : Circuit, store = False):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            store(bool): whether work with local data
        Returns:
            Circuit: output circuit
        """
        gates = circuit.gates
        return solve(gates, circuit.circuit_width(), store)

