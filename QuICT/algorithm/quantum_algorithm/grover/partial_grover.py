#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/6/3 10:14 上午
# @Author  : Peng Sirui
# @File    : partial_grover.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux

from QuICT.simulation.state_vector import ConstantStateVectorSimulator


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
    def run(n, k, oracle, simulator=None):
        """ partial grover search with one target

        Args:
            f(list<int>): the function to be decided
            n(int):       bits length of global address
            k(int):       bits length of block address
            oracle(CompositeGate):   the oracle assuming one ancilla@[n] in |->
        Returns:
            int: the target address, big endian
        """
        if simulator is None:
            simulator = ConstantStateVectorSimulator()
        K = 1 << k
        N = 1 << n
        eps = 1 / K  # can use other epsilon
        r1, r2 = calculate_r1_r2_one_target(N, K, eps)

        circuit = Circuit(n + 3)
        qreg = list(range(n))
        ancilla = n
        dirty = n + 1
        ctarget = n + 2
        cqreg = [n + 2] + [i for i in range(n)]
        # step 1
        for idx in qreg: H | circuit(idx)
        X | circuit(ancilla)
        H | circuit(ancilla)
        for i in range(r1):
            # global inversion about target
            oracle | circuit(qreg + [ancilla])
            # global inversion about average
            for idx in qreg: H | circuit(idx)
            for idx in qreg: X | circuit(idx)
            H | circuit(qreg[n - 1])
            MCTLinearOneDirtyAux.execute(n + 1) | circuit(qreg + [dirty])
            H | circuit(qreg[n - 1])
            for idx in qreg: X | circuit(idx)
            for idx in qreg: H | circuit(idx)
        # step 2
        for i in range(r2):
            # global inversion about target
            oracle | circuit(qreg + [ancilla])
            # local inversion about average
            local_n = n - k
            local_qreg = [j for j in range(k, k + local_n)]
            for idx in local_qreg: H | circuit(idx)
            for idx in local_qreg: X | circuit(idx)
            H | circuit(local_qreg[local_n - 1])
            MCTLinearOneDirtyAux.execute(
                local_n + 1) | circuit(local_qreg + [dirty])
            H | circuit(local_qreg[local_n - 1])
            for idx in local_qreg: X | circuit(idx)
            for idx in local_qreg: H | circuit(idx)
        # step 3
        oracle | circuit(qreg + [ctarget])
        # controlled inversion about average
        CH | circuit([qreg[n - 1]] + [ctarget])
        CH | circuit([qreg[n - 1]] + [ctarget])
        CH | circuit([qreg[n - 1]] + [ctarget])
        MCTLinearOneDirtyAux.execute(
            n + 2) | circuit(cqreg[0:n] + [qreg[n - 1]] + [ancilla])
        CH | circuit([qreg[n - 1]] + [ctarget])
        CH | circuit([qreg[n - 1]] + [ctarget])
        CH | circuit([qreg[n - 1]] + [ctarget])
        # Measure
        for idx in qreg: Measure | circuit(idx)
        simulator.run(circuit)
        return int(circuit[qreg])
