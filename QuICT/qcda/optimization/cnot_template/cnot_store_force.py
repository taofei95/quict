#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/11 1:37 下午
# @Author  : Han Yu
# @File    : cnot_store_force.py

import json
import os

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

        if qubit_chart[n] is None:
            path = f"{os.path.dirname(os.path.abspath(__file__))}{os.path.sep}json \
                {os.path.sep}{n}qubit_cnot.json"
            with open(path, "r") as f:
                json_data = f.readlines()
                qubit_chart[n] = json.loads(json_data)

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

    @staticmethod
    def _run(circuit : Circuit, *pargs):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            *pargs: other parameters
        Returns:
            Circuit: output circuit
        """
        gates = circuit.gates
        return solve(gates)
