#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/2/16 4:25 下午
# @Author  : Han Yu
# @File    : cnot_force_depth.py

import json
import os

from QuICT import *

def count(nn):
    ans = 1
    for i in range(nn):
        ans *= ((1 << nn) - (1 << i))
    return ans

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
    target_col : int = n * target

    for i in range(n):
        if state & (1 << (control_col + i)):
            state ^=  (1 << (target_col + i))
    return state

def generate_json(n):
    """ find the best circuit by bfs

    Args:
        n(int): input qubits

    Returns:
        Circuit: optimal circuit

    """
    now = 0
    for i in range(n):
        now ^= (1 << (i * n + i))
    vis = np.zeros(1 << (n * n), dtype=int)
    vis[now] = 1
    pre = [None] * (1 << (n * n))
    out = {}
    queue = [now]
    out[now] = []
    ans = count(n)
    l = 0
    total = 1
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
                queue.append(new_state)
                paths = []
                total += 1
                while new_state != queue[0]:
                    paths.extend(layers[pre[new_state][1]])
                    new_state = pre[new_state][0]
                out[queue[-1]] = []
                for index in range(len(paths) - 1, -1, -1):
                    out[queue[-1]].append((paths[index]))

        if total == ans:
            break
        l += 1

    keys = out.keys()
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/{n}qubit_cnot_depth.inf", 'w') as file:
        file.write(";")
        for key in keys:
            string = f"{key}:"
            tuples = out[key]
            len_tuples = len(tuples)
            for i in range(len_tuples):
                _tuple = tuples[i]
                string += f"{_tuple[0] * 5 + _tuple[1]}"
                if i + 1 < len_tuples:
                    string += ','
                else:
                    string += ';'
            if len_tuples == 0:
                string += ';'
            file.write(string)

generate_json(2)
generate_json(3)
generate_json(4)
generate_json(5)
