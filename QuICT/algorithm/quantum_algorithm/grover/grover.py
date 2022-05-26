#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:20 上午
# @Author  : Zhu Qinlin
# @File    : standard_grover.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux

from QuICT.simulation.cpu_simulator import CircuitSimulator
import logging


def degree_counterclockwise(v1: np.ndarray, v2: np.ndarray):
    """from v1 to v2
    """
    d = np.real(np.arccos(sum(v1 * v2) / np.sqrt(sum(v1 * v1) * sum(v2 * v2))))
    if d > 0.5 * np.pi:
        d = np.pi - d
    return d


class Grover:
    """ simple grover

    Quantum Computation and Quantum Information - Michael A. Nielsen & Isaac L. Chuang
    """

    @staticmethod
    def run(n, k, oracle, simulator=CircuitSimulator()):
        """ grover search for f with custom oracle

        Args:
            n(int): the length of input of f
            k(int): length of oracle working space. assume clean
            oracle(CompositeGate): the oracle that flip phase of target state.
                [0:n] is index qreg,
                [n:n+k] is ancilla
        Returns:
            int: the a satisfies that f(a) = 1
        """
        assert k > 0, "at least 1 ancilla, which is shared by MCT part"
        circuit = Circuit(n + k)
        index_q = list(range(n))
        ancilla_q = list(range(n, n + k))
        N = 2 ** n
        theta = 2 * np.arccos(np.sqrt(1 - 1 / N))
        T = round(np.arccos(np.sqrt(1 / N)) / theta)

        # create equal superposition state in index_q
        for idx in index_q:
            H | circuit(idx)
        # rotation
        for i in range(T):
            # Grover iteration
            oracle | circuit(index_q + ancilla_q)
            for idx in index_q:
                H | circuit(idx)
            # control phase shift
            for idx in index_q:
                X | circuit(idx)
            H | circuit(index_q[n - 1])
            MCTOneAux.execute(n + 1) | circuit(index_q + ancilla_q[:1])

            H | circuit(index_q[n - 1])
            for idx in index_q:
                X | circuit(idx)
            # control phase shift end
            for idx in index_q:
                H | circuit(idx)
        for idx in index_q:
            Measure | circuit(idx)
        simulator.run(circuit)
        logging.info(f"circuit width          = {circuit.width():4}")
        logging.info(f"circuit depth          = {circuit.depth():4}")
        logging.info(f"circuit size           = {circuit.size():4}")
        # logging.info(f"Grover iteration size  = {oracle_size:4}+{phase_size:4}")
        return int(circuit[index_q])
