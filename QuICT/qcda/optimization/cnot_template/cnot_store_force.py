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


class CnotStoreForceBfs(Optimization):
    """ use bfs to optimize the cnot circuit

    """

    qubit_chart = [None, None, None, None, None]

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
        if CnotStoreForceBfs.qubit_chart[n - 1] is None:
            path = f"{os.path.dirname(os.path.abspath(__file__))}{os.path.sep}json{os.path.sep}{n}qubit_cnot.json"
            if not os.path.exists(path):
                from .json.cnot_bfs import generate_json
                generate_json(n)
            with open(path, "r") as f:
                CnotStoreForceBfs.qubit_chart[n - 1] = ujson.load(f)

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
        ans = CnotStoreForceBfs.qubit_chart[n - 1][str(goal)]
        for plist in ans:
            CX | circuit(plist)
        return circuit

    @staticmethod
    def _run(circuit : Circuit, *pargs):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            *pargs: other parameters
        Returns:
            Circuit: output circuit
        """
        return CnotStoreForceBfs.solve(circuit)
