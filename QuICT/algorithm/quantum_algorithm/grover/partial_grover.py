#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/6/3 10:14 上午
# @Author  : Peng Sirui
# @File    : partial_grover.py

import numpy as np
import logging

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux

from QuICT.simulation.cpu_simulator import CircuitSimulator


def calculate_r1_r2_one_target(N, K):
    # see https://arxiv.org/abs/quant-ph/0504157
    r1 = np.sqrt(N) * np.pi * 0.25 - np.sqrt(N/K) * np.sqrt(3/4)
    r2 = np.sqrt(N/K) * np.pi * (1/6)
    r1 = round(r1)
    r2 = round(r2)
    return r1, r2


class PartialGrover:
    """ partial grover search with one target

    https://arxiv.org/abs/quant-ph/0407122
    """

    @staticmethod
    def circuit(n, n_block, k, oracle, measure=True):
        """ partial grover search with one target

        Args:
            f(list<int>): the function to be decided
            n(int):       bits length of global address
            n_block(int): bits length of block address
            k(int):       length of oracle working space. assume clean
            oracle(CompositeGate): the oracle that flip phase of target state.
                [0:n] is index qreg,
                [n:n+k] is ancilla
                also assume that it works in style of QCQI p249 6.2
        Returns:
            int: the target address, big endian
        """
        assert k >= 2, "at least 2 ancilla, which is shared bt the Grover part"
        K = 1 << n_block
        N = 1 << n
        # eps = 1 / np.sqrt(K)  # can use other epsilon
        r1, r2 = calculate_r1_r2_one_target(N, K)

        circuit = Circuit(n + k + 1)
        index_q = list(range(n))
        oracle_q = list(range(n, n + k))
        ancillia_q = [n + k]
        # step 1
        for idx in index_q:
            H | circuit(idx)
        for i in range(r1):
            # global inversion about target
            oracle | circuit(index_q + oracle_q)
            # global inversion about average
            for idx in index_q:
                H | circuit(idx)
            for idx in index_q:
                X | circuit(idx)
            H | circuit(index_q[n - 1])
            MCTOneAux.execute(n + 1) | circuit(index_q + oracle_q[:1])
            H | circuit(index_q[n - 1])
            for idx in index_q:
                X | circuit(idx)
            for idx in index_q:
                H | circuit(idx)
        # step 2
        for i in range(r2):
            # global inversion about target
            oracle | circuit(index_q + oracle_q)
            # local inversion about average
            local_n = n - n_block
            local_index_q = [j for j in range(n_block, n_block + local_n)]
            for idx in local_index_q:
                H | circuit(idx)
            for idx in local_index_q:
                X | circuit(idx)
            H | circuit(local_index_q[local_n - 1])
            MCTOneAux.execute(local_n + 1) | circuit(local_index_q + oracle_q[:1])
            H | circuit(local_index_q[local_n - 1])
            for idx in local_index_q:
                X | circuit(idx)
            for idx in local_index_q:
                H | circuit(idx)
        # step 3
        H | circuit(ancillia_q[0])
        X | circuit(ancillia_q[0])
        oracle | circuit(index_q + [ancillia_q[0]] + oracle_q[1:])
        X | circuit(ancillia_q[0])
        H | circuit(ancillia_q[0])
        # controlled inversion about average
        for idx in index_q:
            CH | circuit([ancillia_q[0], idx])
        for idx in index_q:
            CX | circuit([ancillia_q[0], idx])
        MCTOneAux.execute(n + 2) | circuit([ancillia_q[0]] + index_q + oracle_q[:1])
        for idx in index_q:
            CX | circuit([ancillia_q[0], idx])
        for idx in index_q:
            CH | circuit([ancillia_q[0], idx])
        # Measure
        for idx in index_q:
            if measure:
                Measure | circuit(idx)
        logging.info(f"circuit width           = {circuit.width():4}")
        # logging.info(f"circuit depth          = {circuit.depth():4}")
        logging.info(f"global Grover iteration = {r1:4}")
        logging.info(f"local  Grover iteration = {r2:4}")
        logging.info(f"oracle  calls           = {r1+r2+1:4}")
        # logging.info(f"oracle  size           = {oracle.size():4}")
        logging.info(f"other circuit size      = {circuit.size() - oracle.size()*(r1+r2):4}")
        return circuit

    @staticmethod
    def run(n, n_block, k, oracle, simulator=CircuitSimulator()):
        index_q = list(range(n))
        circuit = PartialGrover.circuit(n, n_block, k, oracle)
        simulator.run(circuit)
        return int(circuit[index_q])
