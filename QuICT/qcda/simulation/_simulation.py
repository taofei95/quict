#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulation

import numpy as np

from QuICT.core import *
from QuICT.ops.linalg import dot, MatrixPermutation

# tool function
def build_2qubit_gate(compositeGate, unitary, target1, target2):
    if target1 > target2:
        target1, target2 = target2, target1
    ugate = Unitary(unitary).copy()
    ugate.targs = [target1, target2]
    compositeGate.append(ugate)

def build_1qubit_gate(compositeGate, unitary, target1):
    ugate = Unitary(unitary).copy()
    ugate.targs = [target1]
    compositeGate.append(ugate)

# cost function
def tensor_cost(a, b):
    return 1.0 * ((1 << b) ** 2 - (1 << a) ** 2)

def tensor_both_cost(a, b):
    return 1.0 * ((1 << (b + a)) ** 2)

def multiply_cost(k):
    return 1.0 * ((1 << k) ** 3)

def multiply_vector_cost(k):
    return 1.0 * ((1 << k) ** 2)

# tool class
class dp:
    def __init__(self, args, value=0):
        self.set = set(args)
        self.length = len(self.set)
        self.value = value

    def merge(self, other, value=0):
        return dp(self.set | other.set, value)

    def merge_value(self, other):
        k = len(self.set | other.set)
        if len(self.set & other.set) == 0:
            return tensor_both_cost(self.length, other.length)
        return tensor_cost(self.length, k) + tensor_cost(other.length, k) + multiply_cost(k)

    def amplitude_cost(self, width):
        return self.value + tensor_cost(self.length, width) + multiply_vector_cost(width)

class BasicSimulator(object):

    @staticmethod
    def pretreatment(circuit):
        """
        :param
            circuit(Circuit): the circuit needs pretreatment.
        :return:
            CompositeGate: the gates after pretreatment
        """
        gates = CompositeGate()
        circuit_width = circuit.circuit_width()
        gateSet = [np.identity(2, dtype=np.complex) for _ in range(circuit_width)]
        tangle = [i for i in range(circuit.circuit_width())]
        for gate in circuit.gates:
            if gate.target + gate.controls >= 3:
                raise Exception("only support 2-qubit gates and 1-qubit gates.")
            # 1-qubit gate
            if gate.target + gate.controls == 1:
                target = gate.target
                if tangle[target] == target:
                    gateSet[target] = dot(gate.matrix, gateSet[target])
                else:
                    if tangle[target] < target:
                        gateSet[target] = dot(np.kron(np.identity(2, dtype=np.complex), gate.matrix), gateSet[target])
                    else:
                        gateSet[target] = dot(np.kron(gate.matrix, np.identity(2, dtype=np.complex)), gateSet[target])
                    gateSet[tangle[target]] = target
            # 2-qubit gate
            else:
                affectArgs = gate.affectArgs
                target1, target2 = affectArgs[0], affectArgs[1]
                if target1 < target2:
                    matrix = gate.compute_matrix
                else:
                    matrix = MatrixPermutation(gate.compute_matrix, [1, 0])

                if tangle[target1] == target2:
                    gateSet[target1] = dot(matrix, gateSet[target1])
                    gateSet[target2] = gateSet[target1]
                elif tangle[target1] == target1 and tangle[target2] == target2:
                    if target1 < target2:
                        target_matrix = np.kron(gateSet[target1], gateSet[target2])
                    else:
                        target_matrix = np.kron(gateSet[target2], gateSet[target1])
                    gateSet[target1] = dot(matrix, target_matrix)
                    gateSet[target2] = gateSet[target1]
                    tangle[target1], tangle[target2] = target2, target1
                else:
                    if tangle[target1] != target1:
                        revive = target2
                        target = target1
                    else:
                        revive = target1
                        target = target2
                    if not np.isclose(np.identity(4, dtype=np.complex), gateSet[target]):
                        build_2qubit_gate(gates, gateSet[target], target, tangle[target])
                    gateSet[tangle[target]] = np.identity(2, dtype=np.complex)
                    gateSet[target] = np.identity(2, dtype=np.complex)
                    tangle[tangle[target]] = tangle[target]
                    tangle[target] = target

                    if tangle[revive] == revive:
                        if revive >= target1 and revive >= target2:
                            target_matrix = np.kron(gateSet[revive], np.identity(2, dtype=np.complex))
                        else:
                            target_matrix = np.kron(np.identity(2, dtype=np.complex), gateSet[revive])
                        gateSet[target1] = dot(matrix, target_matrix)
                        gateSet[target2] = gateSet[target1]
                    else:
                        target = revive
                        if not np.isclose(np.identity(4, dtype=np.complex), gateSet[target]):
                            build_2qubit_gate(gates, gateSet[target], target, tangle[target])
                        gateSet[tangle[target]] = np.identity(2, dtype=np.complex)
                        gateSet[target] = np.identity(2, dtype=np.complex)
                        tangle[tangle[target]] = tangle[target]
                        tangle[target] = target
        for i in range(circuit_width):
            if tangle[i] == i:
                if not np.isclose(np.identity(2, dtype=np.complex), gateSet[i]):
                    build_1qubit_gate(gates, gateSet[i], i)
            elif tangle[i] > i:
                if not np.isclose(np.identity(4, dtype=np.complex), gateSet[i]):
                    build_2qubit_gate(gates, gateSet[i], i, tangle[i])
        return gates

    @staticmethod
    def unitary_merge_layer(gates: list):
        gate_length = len(gates)
        f = [[None if j != i else dp(gates[i]) for j in range(gate_length)] for i in range(gate_length)]
        pre = [[0 for _ in range(gate_length)] for _ in range(gate_length)]

        for interval in range(1, gate_length):
            for j in range(gate_length - interval):
                pre_temp = j
                pre_value = f[j][j].merge_value(f[j + 1][j + interval])
                for k in range(j + 1, j + interval - 1):
                    new_value = f[j][k].merge_value(f[k + 1][j + interval])
                    if new_value < pre_value:
                        pre_value = new_value
                        pre_temp = k
                f[j][j + interval] = f[j][pre_temp].merge(f[pre_temp + 1][j + interval], pre_value)
                pre[j][j + interval] = pre_temp

        return f, pre

    @staticmethod
    def unitary_pretreatment(circuit):
        small_gates = BasicSimulator.pretreatment(circuit)
        gates = []
        for gate in small_gates:
            gates.append(gate.affectArgs.copy())
        # gates as input
        f, pre = unitary_merge_layer(gates)

        order = []

        def pre_search(l, r):
            if l >= r:
                return
            stick = pre[l][r]
            order.append(stick)
            pre_search(l, stick)
            pre_search(stick + 1, r)

        pre_search(0, gate_length - 1)
        order.reverse()
        return order, f[0][gate_length - 1].value

    @staticmethod
    def vector_pretreatment(circuit):
        small_gates = BasicSimulator.pretreatment(circuit)
        gates = []
        for gate in small_gates:
            gates.append(gate.affectArgs.copy())
        # gates as input
        f, pre = unitary_merge_layer(gates)

        for i in range(gate_length):
            pre_temp = 0
            pre_value = f[0][i].amplitude_cost(width)
            for j in range(i):
                new_value = amplitude_f[j] + f[j + 1][i].amplitude_cost(width)
                if new_value < pre_value:
                    pre_value = new_value
                    pre_temp = j
            amplitude_f.append(pre_value)
            pre_amplitude.append(pre_temp)

        order = []

        def pre_search(l, r):
            if l >= r:
                return
            stick = pre[l][r]
            order.append(stick)
            pre_search(l, stick)
            pre_search(stick + 1, r)

        def pre_amplitude_search(r):
            stick = pre_amplitude[r]
            order.append(-1)
            pre_search(stick, r)
            if stick <= 0:
                return
            pre_amplitude_search(stick)

        pre_amplitude_search(gate_length - 1)
        order.reverse()
        return order, f[0][gate_length - 1].value
