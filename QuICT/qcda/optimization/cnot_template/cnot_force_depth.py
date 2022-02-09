#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/2/16 4:25 下午
# @Author  : Han Yu
# @File    : cnot_force_depth.py

import numpy as np

from .._optimization import Optimization
from QuICT.core import *
from QuICT.core.gate import CX, GateType


def generate_layer(n):
    """ generate combination layer for n qubits(n in [2, 5])

    Args:
        n(int): the qubits of layer, in [2, 5]
    Returns:
        list<list<tuple<int, int>>>: the list of layers
    """
    layers = []

    # single layer
    for i in range(n):
        for j in range(n):
            if i != j:
                layers.append([(i, j)])

    # double layer
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if i != j and i != k and i != l and j != k and j != l and k != l:
                        layers.append([(i, j), (k, l)])

    return layers


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
    target_col: int = n * target

    for i in range(n):
        if state & (1 << (control_col + i)):
            state ^= (1 << (target_col + i))
    return state


def solve(input: Circuit):
    """ find the best circuit by bfs

    Args:
        input(Circuit): input circuit

    Returns:
        Circuit: optimal circuit

    """
    n = input.width()
    circuit = Circuit(n)
    input_matrix = np.identity(n, dtype=bool)
    now = 0
    goal = 0
    for gate in input.gates:
        if gate.type != GateType.cx:
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

    layers = generate_layer(n)

    while True:
        now = queue[l]
        for i in range(len(layers)):
            layer = layers[i]
            for cx in layer:
                new_state = apply_cx(now, cx[0], cx[1], n)
            if vis[new_state] == 0:
                pre[new_state] = (now, i)
                vis[new_state] = vis[now] + 1
                if new_state == goal:
                    paths = []
                    while new_state != queue[0]:
                        paths.extend(layers[pre[new_state][1]])
                        new_state = pre[new_state][0]
                    for index in range(len(paths) - 1, -1, -1):
                        CX | circuit(list(paths[index]))

                    return circuit
                queue.append(new_state)
        l += 1


class CnotForceDepthBfs(Optimization):
    """ use bfs to optimize the cnot circuit

    """
    @staticmethod
    def execute(circuit: Circuit, *pargs):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            *pargs: other parameters
        Returns:
            Circuit: output circuit
        """
        return solve(circuit)
