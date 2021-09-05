#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/6/3 10:14 上午
# @Author  : Peng Sirui
# @File    : partial_grover.py

import numpy as np

from .._algorithm import Algorithm
from QuICT import *
from QuICT.qcda.synthesis.mct import MCTLinearOneDirtyAux
    
def calculate_r1_r2_one_target(N, K, eps):
    r1 = np.sqrt(N)*np.pi*0.25*(1-eps)
    r1 = round(r1)
    o_theta = 2*np.arccos(np.sqrt(1-1/N))
    theta = np.pi/2 - (0.5+r1)*o_theta
    sin_theta = np.sin(theta)
    sqrt_K_mul_alpha_yt = np.sqrt(K-sin_theta*sin_theta*(K-1))
    r2 = (np.sqrt(N/K)*0.5)*(np.arcsin(sin_theta/sqrt_K_mul_alpha_yt) +
                             np.arcsin(sin_theta*(K-2)/(2*sqrt_K_mul_alpha_yt)))
    r2 = round(r2)
    return r1, r2

def run_partial_grover(f, n, k, oracle):
    """ partial grover search with one target

    Args:
        f(list<int>): the function to be decided
        n(int):       bits length of global address
        k(int):       bits length of block address
        oracle(function):   the oracle
    Returns:
        int: the target address, big endian
    """
    K = 1 << k
    N = 1 << n
    eps = 1/K  # can use other epsilon
    r1, r2 = calculate_r1_r2_one_target(N, K, eps)

    circuit = Circuit(n + 3)
    qreg = circuit([i for i in range(n)])
    ancilla = circuit(n)
    dirty = circuit(n+1)
    ctarget = circuit(n+2)
    cqreg = circuit([n+2]+[i for i in range(n)])
    # step 1
    H | qreg
    X | ancilla
    H | ancilla
    for i in range(r1):
        # global inversion about target
        oracle(f, qreg, ancilla)
        # global inversion about average
        H | qreg
        X | qreg
        H | qreg(n - 1)
        MCTLinearOneDirtyAux.execute(
            n + 1) | (qreg([j for j in range(0, n - 1)]), qreg(n - 1), dirty)
        H | qreg(n - 1)
        X | qreg
        H | qreg
    # step 2
    for i in range(r2):
        # global inversion about target
        oracle(f, qreg, ancilla)
        # local inversion about average
        local_n = n-k
        local_qreg = qreg([j for j in range(k, k+local_n)])
        H | local_qreg
        X | local_qreg
        H | local_qreg(local_n - 1)
        MCTLinearOneDirtyAux.execute(
            local_n + 1) | (local_qreg([j for j in range(0, local_n - 1)]), local_qreg(local_n - 1), dirty)
        H | local_qreg(local_n - 1)
        X | local_qreg
        H | local_qreg
    # step 3
    oracle(f, qreg, ctarget)
    # controlled inversion about average
    CH | (qreg, ctarget)
    CX | (qreg, ctarget)
    CH | (qreg(n - 1), ctarget)
    MCTLinearOneDirtyAux.execute(
        n + 2) | (cqreg([j for j in range(0, n)]), qreg(n - 1), ancilla)
    CH | (qreg(n - 1), ctarget)
    CX | (qreg, ctarget)
    CH | (qreg, ctarget)
    # Measure
    Measure | qreg
    circuit.exec()
    return int(qreg)

class PartialGrover(Algorithm):
    """ partial grover search with one target

    https://arxiv.org/abs/quant-ph/0407122
    """
    @classmethod
    def run(cls, f, n, k, oracle):
        """ partial grover search with one target

        Args:
            f(list<int>): the function to be decided
            n(int):       bits length of global address
            k(int):       bits length of block address
            oracle(function):   the oracle
        Returns:
            int: the target block address
        """
        assert len(f) == (1 << n)
        return run_partial_grover(f, n, k, oracle)
