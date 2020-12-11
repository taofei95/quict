#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/5/19 10:58 上午
# @Author  : Han Yu
# @File    : Shor.py

from fractions import Fraction
import random

import numpy as np

from .._algorithm import Algorithm
from QuICT.core import *

def EX_GCD(a, b, arr):
    if b == 0:
        arr[0] = 1
        arr[1] = 0
        return a
    g = EX_GCD(b, a % b, arr)
    t = arr[0]
    arr[0] = arr[1]
    arr[1] = t - int(a / b) * arr[1]
    return g

def ModReverse(a, n):
    arr = [0, 1]
    EX_GCD(a, n, arr)
    return (arr[0] % n + n) % n

def fast_power(a, b, N):
    x = 1
    now_a = a
    while b > 0:
        if b % 2 == 1:
            x = x * now_a % N
        now_a = now_a * now_a % N
        b >>= 1
    return x

def classical_cUa(a, N, L, circuit):
    plist = [0]
    for i in range(1, L + 1):
        plist.append(i)
    ControlPermMulDetail([a, N]) | circuit(plist)

def Shor(N, fidelity = None):
    """ run the algorithm with fidelity
    Args:
        N(int): the number to be factored
        fidelity(float): the fidelity
    Returns:
        int: a factor of n
        int: the base number a
        int: the period of a for N
        int: the round shor run
        list<float>: the probability of the first register
    """
    # 1. If N is even, return the factor 2
    if N % 2 == 0:
        return 2, 0, 0, 0, []

    # 2. Classically determine if N = p^q
    y = np.log2(N)
    L = int(np.ceil(np.log2(N)))
    for b in range(3, L):
        x = y / b
        squeeze = np.power(2, x)
        u1 = int(np.floor(squeeze))
        u2 = int(np.ceil(squeeze))
        if fast_power(u1, b, N) == 0 or fast_power(u2, b, N) == 0:
            return b, 0, 0, 0, []

    rd = 0
    while True:
        # 3. Choose a random number a, 1 < a <= N - 1
        a = random.randint(2, N - 1)
        gcd = np.gcd(a, N)
        if gcd > 1:
            return gcd, 0, 0, rd, []
        rd += 1

        # 4. Use the order-finding quantum algorithm to find the order r of a modulo N
        NN = N
        Nan = []
        for i in range(L + 1):
            Nan.append(NN % 2)
            NN >>= 1
        Nan.reverse()
        Nth = []
        for i in range(L + 1):
            tht = 0.0
            coe = 1.0
            for j in range(i, L + 1):
                if Nan[j] == 1:
                    tht += coe
                coe /= 2
            Nth.append(tht)
        Nth.reverse()
        # 1
        # L
        circuit = Circuit(L + 1)
        # circuit.reset_initial_zeros()
        if fidelity is not None:
            circuit.fidelity = fidelity
        X | circuit(1)
        a_r = ModReverse(a, N)
        aa = a
        aa_r = a_r
        Rth = 0
        M = 0.0
        a_list = []
        a_r_list = []
        for i in range(2 * L):
            a_list.append(aa)
            a_r_list.append(aa_r)
            aa = aa * aa % N
            aa_r = aa_r * aa_r % N
        a_list.reverse()
        a_r_list.reverse()
        for i in range(2 * L):
            H | circuit(0)

            aa = a_list[i]
            classical_cUa(aa, N, L, circuit)
            if i != 0:
                Rz(-np.pi * Rth) | circuit(0)
            H | circuit(0)

            Measure | circuit(0)

            circuit.exec_release()

            measure = int(circuit(0))
            if measure == 1:
                Rth += 1
                X | circuit(0)
                M += 1.0 / (1 << (2 * L - i))
            Rth /= 2

        r = Fraction(M).limit_denominator(N - 1).denominator

        # 5. cal
        if fast_power(a, r, N) != 1 or r % 2 == 1 or fast_power(a, r // 2, N) == N - 1:
            continue
        b = np.gcd(fast_power(a, r // 2, N) - 1, N)
        if N % b == 0 and b != 1 and N != b:
            return b, a, r, rd, []
        c = np.gcd(fast_power(a, r // 2, N) + 1, N)
        if N % c == 0 and c != 1 and N != b:
            return b, a, r, rd, []

class classical_zip_shor_factor(Algorithm):
    """ shor algorithm with oracle decomposed into gates.
        Use classical method calculate the oracle, first register zip to 1.

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3

    a (L+1)-bit number need (2L) qubits

    """
    @staticmethod
    def _run(n, fidelity = None):
        """ run the algorithm with fidelity
        Args:
            n(int): the number to be factored
            fidelity(float): the fidelity
        Returns:
            int: a factor of n
            int: the base number a
            int: the period of a for N
            int: the round shor run
            list<float>: the probability of the first register
        """
        return Shor(n, fidelity)
