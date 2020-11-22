#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/22 7:14 下午
# @Author  : Han Yu
# @File    : cnot_force.py

import numpy as np

from .._optimization import Optimization
from QuICT.models import *

class path(object):
    """ record the path of bfs

    Attribute:
        father_node(int): father in bfs
        CX_tuple(tuple<int, int>): the way father access to son
    """

    def __init__(self, father_node, control, target):
        """ initial method

        father_node(int): the order of father_node
        control(int): the control bit of CX
        target(int): the target bit of CX
        """
        self.father_node = father_node
        self.CX_tuple = (control, target)

def apply_cx(state, control, target, n):
    """ apply cnot gate to the state

    Args:
        state(int): the state represent the matrix
        control(int): the control index for the cx gate
        target(int): the target index for the cx gate
        n(int): number of qubits in the matrix
    Returns:
        int: the new state after applying the gate
    """

    control_col: int = n * control
    target_col : int = n * target

    for i in range(n):
        if state & (1 << (control_col + i)):
            state ^=  (1 << (target_col + i))
    return state

def solve(input: Circuit):
    """ find the best circuit by bfs

    Args:
        input(Circuit): input circuit

    Returns:
        Circuit: optimal circuit

    """
    n = input.circuit_length()
    circuit = Circuit(n)
    input_matrix = np.identity(n, dtype=bool)
    now = 0
    goal = 0
    for gate in input.gates:
        if gate.type() != GateType.CX:
            raise Exception("the circuit should only contain CX gate")
        input_matrix[gate.targ, :] = input_matrix[gate.targ, :] ^ input_matrix[gate.carg, :]
    for i in range(n):
        for j in range(n):
            if input_matrix[i, j]:
                goal ^= 1 << (i * n + j)
    for i in range(n):
        now ^= (1 << (i * n + i))
    vis = np.zeros(1 << (n * n), dtype=int)
    vis[now] = 1
    pre = [None] * (1 << (n * n))
    queue = [now]
    l = 0
    if now == goal:
        return circuit

    while True:
        now = queue[l]
        for i in range(n):
            for j in range(n):
                if i != j:
                    new_state = apply_cx(now, i, j, n)
                    if vis[new_state] == 0:
                        pre[new_state] = path(now, i, j)
                        vis[new_state] = vis[now] + 1
                        if new_state == goal:
                            paths = []
                            while new_state != queue[0]:
                                paths.append(pre[new_state].CX_tuple)
                                new_state = pre[new_state].father_node
                            for index in range(len(paths) - 1, -1, -1):
                                CX | circuit(paths[index])

                            return circuit
                        queue.append(new_state)
        l += 1
        if l % 100 == 0:
            print(l)


class cnot_force_bfs(Optimization):
    """ use bfs to optimize the cnot circuit

    """
    @staticmethod
    def _run(circuit : Circuit, *pargs):
        """
        circuit(Circuit): the circuit to be optimize
        *pargs: other parameters
        """
        return solve(circuit)
