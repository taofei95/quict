#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:20 上午
# @Author  : Zhu Qinlin
# @File    : standard_grover.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux
from QuICT.algorithm import amplitude

from QuICT.algorithm.amplitude.amplitude import Amplitude
import logging


def my_print(msg, demo_mode):
    if demo_mode:
        print(msg)
    else:
        logging.info(msg)


def degree_counterclockwise(v1: np.ndarray, v2: np.ndarray):
    """from v1 to v2
    """
    d = np.real(np.arccos(sum(v1*v2)/np.sqrt(sum(v1*v1)*sum(v2*v2))))
    if d > 0.5*np.pi:
        d = np.pi-d
    return d


class Grover:
    """ simple grover

    Quantum Computation and Quantum Information - Michael A. Nielsen & Isaac L. Chuang
    """
    @staticmethod
    def run(f, n, oracle, demo_mode=False, **kwargs):
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
        my_print(
            f"[init] theta = {theta:.4f}, Grover iteration count = {T}", demo_mode)
        phase_size = 0
        oracle_size = 0

        # create equal superposition state in index_q
        H | index_q
        # create |-> in result_q
        X | result_q
        H | result_q
        for i in range(T):
            # Grover iteration
            if demo_mode and i==0:
                tmp = circuit.circuit_size()
            oracle(f, index_q, result_q)
            if demo_mode and i==0:
                oracle_size = circuit.circuit_size() - tmp
                tmp = circuit.circuit_size()
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
            if demo_mode and i==0:
                phase_size = circuit.circuit_size() - tmp
            if demo_mode:
                amp = Amplitude.run(circuit)
                amp = np.array(amp[::2])*np.sqrt(2)
                d = degree_counterclockwise(amp, kwargs["beta"])
                my_print(
                    f"[{i+1:3}-th Grover iteration] "
                    +f"degree from target state: {d:.3f} "
                    +f"success rate:{(np.real(amp[kwargs['target']])**2)*100:.1f}%", demo_mode)
        amp = Amplitude.run(circuit)
        if demo_mode:
            amp = np.array(amp[::2])*np.sqrt(2)
        Measure | index_q
        circuit.exec()
        my_print(f"circuit width          = {circuit.circuit_width():4}", demo_mode)
        my_print(f"circuit depth          = {circuit.circuit_depth():4}", demo_mode)
        my_print(f"circuit size           = {circuit.circuit_size():4}", demo_mode)
        my_print(f"Grover iteration size  = {oracle_size:4}+{phase_size:4}", demo_mode)
        return int(index_q)
