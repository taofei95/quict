#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/11 1:37 下午
# @Author  : Han Yu
# @File    : cnot_store_force.py

import ujson
import os
import time

from .._optimization import Optimization
from QuICT.core import *


class CnotStoreForceDepthBfs(Optimization):
    """ use bfs to optimize the cnot circuit

    """
    @staticmethod
    def solve(input: Circuit):
        """ find the best circuit by bfs

        Args:
            input(Circuit): input circuit

        Returns:
            Circuit: optimal circuit

        """
        n = input.circuit_width()
        if n > 5:
            raise Exception("the qubits number should be smaller than or equal to 5.")
        path = f"{os.path.dirname(os.path.abspath(__file__))}{os.path.sep}json{os.path.sep}{n}qubit_cnot_depth.inf"
        if not os.path.exists(path):
            from .json.cnot_depth_bfs import generate_json
            generate_json(n)
        with open(path, "r") as f:
            loadnow = f.readline()

        circuit = Circuit(n)
        input_matrix = np.identity(n, dtype=bool)
        goal = 0
        for gate in input.gates:
            if gate.type() != GATE_ID["CX"]:
                raise Exception("the circuit should only contain CX gate")
            input_matrix[gate.targ, :] = input_matrix[gate.targ, :] ^ input_matrix[gate.carg, :]
        for i in range(n):
            for j in range(n):
                if input_matrix[i, j]:
                    goal ^= 1 << (i * n + j)

        goal_string = f";{goal}:"
        index = loadnow.find(goal_string)
        if index == -1:
            raise Exception("generate error")
        begin = index + len(goal_string)
        end = loadnow.find(";", begin)
        tuples_encode = loadnow[begin:end].split(",")
        ans = []
        for tuple_encode in tuples_encode:
            if len(tuple_encode) > 0:
                ans.append((int(tuple_encode) // 5, int(tuple_encode) % 5))
        for plist in ans:
            CX | circuit(plist)
        return circuit

    @staticmethod
    def execute(circuit : Circuit, *pargs):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            *pargs: other parameters
        Returns:
            Circuit: output circuit
        """
        return CnotStoreForceDepthBfs.solve(circuit)