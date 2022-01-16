#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:20 上午
# @Author  : Zhu Qinlin
# @File    : standard_grover.py

import numpy as np

from QuICT import *
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux
from QuICT.algorithm import amplitude

from quict.QuICT.algorithm.amplitude.amplitude import Amplitude


class Grover:
    """ simple grover

    Quantum Computation and Quantum Information - Michael A. Nielsen & Isaac L. Chuang
    """
    @staticmethod
    def run(f, n, oracle):
        """ grover search for f with custom oracle

        Args:
            f(list<int>): the function to be decided
            n(int): the length of input of f
            oracle(function): the oracle
        Returns:
            int: the a satisfies that f(a) = 1
        """
        circuit = Circuit(n + 1)
        index_q = circuit([i for i in range(n)])
        result_q = circuit(n)
        N = 2**n
        theta = 2 * np.arccos(np.sqrt(1 - 1 / N))
        T = round(np.arccos(np.sqrt(1 / N)) / theta)

        # create equal superposition state in index_q
        H | index_q
        # create |-> in result_q
        X | result_q
        H | result_q
        for i in range(T):
            # Grover iteration
            oracle(f, index_q, result_q)
            H | index_q
            # control phase shift
            X | index_q
            H | index_q(n - 1)
            MCTLinearOneDirtyAux.execute(
                n + 1) | (index_q([j for j in range(0, n - 1)]), index_q(n - 1), result_q)
            H | index_q(n - 1)
            X | index_q
            # control phase shift end
            H | index_q
        Amplitude.run(circuit)
        Measure | index_q
        circuit.exec()
        return int(index_q)
