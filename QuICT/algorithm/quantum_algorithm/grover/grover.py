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

    def __init__(self, n, n_ancilla, oracle, n_solution=1, measure=True, simulator=CircuitSimulator()) -> None:
        """
        Args:
            n(int): the length of input of f
            n_ancilla(int): length of oracle working space. assume clean
            oracle(CompositeGate): the oracle that flip phase of target state.
                [0:n] is index qreg,
                [n:n+k] is ancilla
            n_solution(int): number of solution 
            measure(bool): measure included or not
        """
        assert n_ancilla > 0, "at least 1 ancilla, which is shared by MCT part"
        self.n = n
        self.n_ancilla = n_ancilla
        self.oracle = oracle
        self.n_solution = n_solution
        self.measure = measure
        self.simulator = simulator

    def circuit(self):
        """ grover search for f with custom oracle

        Returns:
            int: the a satisfies that f(a) = 1
        """
        n = self.n
        n_ancilla = self.n_ancilla
        oracle = self.oracle
        n_solution = self.n_solution
        measure = self.measure
        circuit = Circuit(n + n_ancilla)
        index_q = list(range(n))
        ancilla_q = list(range(n, n + n_ancilla))
        N = 2 ** n
        theta = 2 * np.arccos(np.sqrt(1 - n_solution / N))
        T = int(np.arccos(np.sqrt(n_solution / N)) / theta)+1

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
            if measure:
                Measure | circuit(idx)
        logging.info(
            f"circuit width          = {circuit.width():4}" +
            f"oracle  calls          = {T:4}" +
            f"other circuit size     = {circuit.size() - oracle.size()*T:4}"
        )
        return circuit

    def run(self):
        index_q = list(range(self.n))
        circuit = self.circuit()
        self.simulator.run(circuit)
        return int(circuit[index_q])
