#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/6/3 10:14 上午
# @Author  : Peng Sirui
# @File    : partial_grover.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux

from QuICT.simulation.cpu_simulator import CircuitSimulator


def calculate_r1_r2_one_target(N, K, eps):
    r1 = np.sqrt(N) * np.pi * 0.25 * (1 - eps)
    r1 = round(r1)
    o_theta = 2 * np.arccos(np.sqrt(1 - 1 / N))
    theta = np.pi / 2 - (0.5 + r1) * o_theta
    sin_theta = np.sin(theta)
    sqrt_K_mul_alpha_yt = np.sqrt(K - sin_theta * sin_theta * (K - 1))
    r2 = (np.sqrt(N / K) * 0.5) * (np.arcsin(sin_theta / sqrt_K_mul_alpha_yt) +
                                   np.arcsin(sin_theta * (K - 2) / (2 * sqrt_K_mul_alpha_yt)))
    r2 = round(r2)
    return r1, r2


class PartialGrover:
    """ partial grover search with one target

    https://arxiv.org/abs/quant-ph/0407122
    """
    @staticmethod
    def run(n, n_block, k, oracle, simulator=CircuitSimulator()):
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
        assert k>=2, "at least 2 ancilla, which is shared bt the Grover part"
        K = 1 << n_block
        N = 1 << n
        eps = 1 / K  # can use other epsilon
        r1, r2 = calculate_r1_r2_one_target(N, K, eps)

        circuit = Circuit(n + k + 1)
        index_q = list(range(n))
        oracle_q = list(range(n,n+k))
        ancillia_q = [n+k]
        # step 1
        for idx in index_q: H | circuit(idx)
        for i in range(r1):
            # global inversion about target
            oracle | circuit(index_q + oracle_q)
            # global inversion about average
            for idx in index_q: H | circuit(idx)
            for idx in index_q: X | circuit(idx)
            H | circuit(index_q[n - 1])
            MCTOneAux.execute(n + 1) | circuit(index_q + oracle_q[:1])
            H | circuit(index_q[n - 1])
            for idx in index_q: X | circuit(idx)
            for idx in index_q: H | circuit(idx)
        # step 2
        for i in range(r2):
            # global inversion about target
            oracle | circuit(index_q + oracle_q)
            # local inversion about average
            local_n = n - n_block
            local_index_q = [j for j in range(n_block, n_block + local_n)]
            for idx in local_index_q: H | circuit(idx)
            for idx in local_index_q: X | circuit(idx)
            H | circuit(local_index_q[local_n - 1])
            MCTOneAux.execute(local_n + 1) | circuit(local_index_q + oracle_q[:1])
            H | circuit(local_index_q[local_n - 1])
            for idx in local_index_q: X | circuit(idx)
            for idx in local_index_q: H | circuit(idx)
        # step 3
        H | circuit(ancillia_q[0])
        X | circuit(ancillia_q[0])
        oracle | circuit(index_q + [ancillia_q[0]] + oracle_q[1:])
        X | circuit(ancillia_q[0])
        H | circuit(ancillia_q[0])
        # controlled inversion about average
        for idx in index_q: CH | circuit([ancillia_q[0], idx])
        for idx in index_q: CX | circuit([ancillia_q[0], idx])
        MCTOneAux.execute(n + 2) | circuit([ancillia_q[0]] + index_q + oracle_q[:1])
        for idx in index_q: CX | circuit([ancillia_q[0], idx])
        for idx in index_q: CH | circuit([ancillia_q[0], idx])
        # Measure
        for idx in index_q: Measure | circuit(idx)
        simulator.run(circuit)
        return int(circuit[index_q])
