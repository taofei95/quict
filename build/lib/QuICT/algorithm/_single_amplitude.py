#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/4/22 1:07 上午
# @Author  : Han Yu
# @File    : _single_amplitude.py

from ._circuit2param import circuit2param, Amplitude
from QuICT.models import *

class circuit_layer(object):
    def __init__(self):
        self.occupy = set()
        self.gates = []

    def addGate(self, gate : BasicGate) -> bool:
        Q_set = set(gate.cargs) | set(gate.targs)
        if len(Q_set & self.occupy) > 0:
            return False
        self.occupy |= Q_set
        self.gates.append(gate)
        return True

    def checkGate(self, gate : BasicGate) -> bool:
        Q_set = set(gate.cargs) | set(gate.targs)
        if len(Q_set & self.occupy) > 0:
            return False
        return True

class f_variable(object):
    def __init__(self, index, f):
        self.variable = index
        self.f = f

    def contract(self):
        return self

def de_layer(gates):
    layers = [circuit_layer()]
    for gate in gates:
        for i in range(len(layers) - 1, -2, -1):
            if i == -1 or not layers[i].checkGate(gate):
                if i + 1 >= len(layers):
                    layers.append(circuit_layer())
                layers[i + 1].addGate(gate)
                break
    return layers

def solve_tree(all_var, functions):
    return 0

def solve(circuit : Circuit, position, initial):
    layers = de_layer(circuit.gates)
    length = circuit.circuit_length()
    v_total = length
    functions = [f_variable([i], [1, 1]) for i in range(length)]
    v_variable = [i for i in range(length)]
    for layer in layers:
        for gate in layer.gates:
            if gate.is_single():
                matrix = gate.matrix
                if gate.is_diagonal():
                    functions.append(f_variable(
                        [v_variable[gate.targ]],
                        [matrix[0], matrix[3]]
                    ))
                else:
                    functions.append(f_variable(
                        [v_variable[gate.targ], v_total],
                        matrix.tolist()
                    ))
                    v_variable[gate.targ] = v_total
                    v_total += 1
            elif gate.is_control_single():
                if gate.is_diagonal():
                    matrix = gate.matrix
                    functions.append(f_variable(
                        [v_variable[gate.carg], v_variable[gate.targ]],
                        [1, matrix[0], 1, matrix[3]]
                    ))
                else:
                    functions.append(f_variable(
                        [v_variable[gate.carg], v_variable[gate.targ], v_total, v_total + 1],
                        gate.compute_matrix.tolist()
                    ))
                    v_variable[gate.carg] = v_total
                    v_total += 1
                    v_variable[gate.targ] = v_total
                    v_total += 1
            elif gate.is_swap():
                functions.append(f_variable(
                    [v_variable[gate.carg], v_variable[gate.targ], v_total, v_total + 1],
                    gate.compute_matrix.tolist()
                ))
                v_variable[gate.carg] = v_total
                v_total += 1
                v_variable[gate.targ] = v_total
                v_total += 1
            else:
                raise Exception("只能给出基础单比特门或基础二比特门")
    v_value = [-1 for _ in range(v_total)]
    for i in range(length):
        if (1 << i) & initial > 0:
            v_value[i] = 1
        else:
            v_value[i] = 0
        if (1 << i) & position > 0:
            if v_variable[i] == i and v_value[i] == 0:
                return 0
            v_value[v_variable[i]] = 1
        else:
            if v_variable[i] == i and v_value[i] == 1:
                return 0
            v_value[v_variable[i]] = 0
    factor = 1
    check_function = []
    for function in functions:
        variable = []
        var = set()
        for i in function.variable:
            if v_value[function.variable[i]] == -1:
                variable.append(i)
                var.add(function.variable[i])
        if len(variable) == 0:
            factor *= sum(function.f)
        else:
            f = function.contract()
            check_function.append(f)
    all_var = set()
    for i in range(v_total):
        all_var.add(i)
    var = set()
    for i in range(length):
        var.add(i)
        var.add(v_variable[i])
    all_var -= var
    return solve_tree(all_var, check_function)

class single_amplitude(circuit2param):
    @staticmethod
    def __run__(circuit : Circuit, position = 0, initial = 0):
        circuit.reset_initial_values(initial)
        # return Amplitude.run(circuit)[position]
        return solve(circuit, position, initial)
