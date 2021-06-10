#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/11 12:26 下午
# @Author  : Han Yu
# @File    : cnot_bfs.py

import json
import os

from QuICT import *

def count(nn):
    ans = 1
    for i in range(nn):
        ans *= ((1 << nn) - (1 << i))
    return ans

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


def generate_json(n):
    """ find the best circuit by bfs

    Args:
        n(int): input qubits
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
    l = 0
    ans = count(n)
    total = 1
    while True:
        now = queue[l]
        for i in range(n):
            for j in range(n):
                if i != j:
                    new_state = apply_cx(now, i, j, n)
                    if vis[new_state] == 0:
                        pre[new_state] = path(now, i, j)
                        vis[new_state] = vis[now] + 1
                        queue.append(new_state)
                        paths = []
                        total += 1
                        while new_state != queue[0]:
                            paths.append(pre[new_state].CX_tuple)
                            new_state = pre[new_state].father_node
                        out[queue[-1]] = []
                        for index in range(len(paths) - 1, -1, -1):
                            out[queue[-1]].append((paths[index]))
        if total == ans:
            break
        l += 1
    # json_data = json.dumps(out)
    # with open("./json/" + str(n) + 'qubit_cnot.json', 'w+') as file:
    #    file.write(json_data)
    keys = out.keys()
    with open(f"{os.path.dirname(os.path.abspath(__file__))}/{n}qubit_cnot.inf", 'w') as file:
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

if __name__ == "__main__":
    for n in range(1, 6):
        generate_json(n)
