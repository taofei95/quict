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

def _matrix_product_to_bigger(space, gate) -> np.ndarray:
    q_len = len(space)
    n = 1 << len(space)

    new_values = np.zeros((n, n), dtype=np.complex128)
    targs = gate.targs
    cargs = gate.cargs
    if not isinstance(targs, list):
        targs = [targs]
    if not isinstance(cargs, list):
        cargs = [cargs]
    circuit_targs = np.append(np.array(cargs, dtype=int).ravel(), np.array(targs, dtype=int).ravel())
    circuit_targs = list(circuit_targs)
    xor = (1 << q_len) - 1
    if not isinstance(targs, list):
        raise Exception("unknown error")

    targs = []
    for number in circuit_targs:
        for i in range(len(space)):
            if space[i] == number:
                targs.append(i)
                break

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
            new_values[i][j] = matrix[nowi][nowj]
    return new_values

def commutative(gateA, gateB):
    spaceA = set()
    spaceB = set()
    for c in gateA.cargs:
        spaceA.add(c)
    for t in gateA.targs:
        spaceA.add(t)
    for c in gateB.cargs:
        spaceB.add(c)
    for t in gateB.targs:
        spaceB.add(t)
    if len(spaceA & spaceB) == 0:
        return True
    space = list(spaceA | spaceB)
    matrix1 = _matrix_product_to_bigger(space, gateA)
    matrix2 = _matrix_product_to_bigger(space, gateB)
    return not np.any(np.abs(matrix1 - matrix2) > 1e-10)

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
        if gate.type() == GATE_ID["CX"]:
            fix_in = int(gate.carg in fix) + int(gate.targ in fix)
            if fix_in == 2:
                local_list.append(gate)
                continue
            elif fix_in == 0:
                output.append(gate)
                continue
        else:
            commu = True
            for s in local_list:
                if not commutative(s, gate):
                    commu = False
                    break
            if commu:
                output.append(gate)
                continue
        if len(local_list) != 0:
            circuit = Circuit(len(fix))
            for local_gate in local_list:
                CX | circuit([mapping[local_gate.carg], mapping[local_gate.targ]])
            new_circuit = (CnotStoreForceBfs if store else CnotForceBfs).execute(circuit)
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
        new_circuit = (CnotStoreForceBfs if store else CnotForceBfs).execute(circuit)
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
    def execute(circuit : Circuit, store = False):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            store(bool): whether work with local data
        Returns:
            Circuit: output circuit
        """
        gates = circuit.gates
        return solve(gates, circuit.circuit_width(), store)

