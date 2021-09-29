#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/29 15:28
# @Author  : Zhu Qinlin
# @File    : HRS_shor.py

'''
The (2n+2)-qubit circuit used in the Shor algorithm is designed by \
THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE \
in "Factoring using 2n+2 qubits with Toffoli based modular multiplication\
'''

import random
import logging
from math import pi
import numpy as np
from fractions import Fraction

from QuICT.core import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from QuICT.algorithm import Algorithm
from .utility import *


def order_finding(a, N):
    """
    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N)))
    t = 2 * n
    logging.info(f'order_finding begin: circuit: L = {n} t = {t}')
    trickbit_store = [0] * t
    circuit = Circuit(2 * n + 3)
    x_reg = circuit([i for i in range(n + 1, 2 * n + 1)])
    # ancilla = circuit([i for i in range(n+1)])
    trickbit = circuit(2 * n + 1)
    # qreg_low= circuit(2 * n + 2)
    X | x_reg[n - 1]
    for k in range(t):
        H | trickbit
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        BEACUa.execute(n, gate_pow, N) | circuit
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | trickbit
        H | trickbit

        Measure | trickbit
        circuit.exec()
        logging.info(f'the {k}th trickbit measured to be {int(trickbit)}')
        trickbit_store[k] = int(trickbit)
        if trickbit_store[k] == 1:
            X | trickbit
    Measure | x_reg
    trickbit_store.reverse()
    logging.info(f'phi~ (approximately s/r) in binary form is {trickbit_store}')

    # continued fraction procedure
    phi_ = sum([(trickbit_store[i] * 1. / (1 << (i + 1))) for i in range(t)])
    logging.info(f'phi~ (approximately s/r) in decimal form is {phi_}')
    if phi_ == 0.0:
        logging.info('order_finding failed: phi~ = 0')
        return 0
    (num, den) = (Fraction(phi_).numerator, Fraction(phi_).denominator)
    CFE = continued_fraction_expansion(num, den)
    logging.info(f'Continued fraction expansion of phi~ is {CFE}')
    num1 = CFE[0]
    den1 = 1
    num2 = 1
    den2 = 0
    logging.info(f'the 0th convergence is {num1}/{den1}')
    for k in range(1, len(CFE)):
        num = num1 * CFE[k] + num2
        den = den1 * CFE[k] + den2
        logging.info(f'the {k}th convergence is {num}/{den}')
        if den >= N:
            break
        else:
            num2 = num1
            num1 = num
            den2 = den1
            den1 = den
    r = den1
    if pow(a, r, N) == 1:
        logging.info(f'order_finding succeed: r = {r} is the order of a = {a}')
        return r
    else:
        logging.info(f'order_finding failed: r = {r} is the order of a = {a}')
        return 0


class BEAShorFactor(Algorithm):
    '''
    The (2n+2)-qubit circuit used in the Shor algorithm is designed by \
    THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE \
    in "Factoring using 2n+2 qubits with Toffoli based modular multiplication\
    '''
    @staticmethod
    def run(N):
        # check if input is prime (using MillerRabin in klog(N), k is the number of rounds to run MillerRabin)
        assert (not miller_rabin(N)), 'N is prime'

        # 1. If n is even, return the factor 2
        if N % 2 == 0:
            logging.info('Shor succeed: N is even, found factor 2 classically')
            return 2

        # 2. Classically determine if N = p^q
        y = np.log2(N)
        L = int(np.ceil(np.log2(N)))
        for b in range(2, L):
            x = y / b
            squeeze = np.power(2, x)
            u1 = int(np.floor(squeeze))
            u2 = int(np.ceil(squeeze))
            if pow(u1, b) == N:
                logging.info(
                    f'Shor succeed: N is exponential, found the only factor {u1} classically')
                return u1
            if pow(u2, b) == N:
                logging.info(
                    f'Shor succeed: N is exponential, found the only factor {u2} classically')
                return u2

        rd = 0
        max_rd = 15
        while rd < max_rd:
            # 3. Choose a random number a (1<a<N)
            a = random.randint(2, N - 1)
            gcd = np.gcd(a, N)
            if gcd > 1:
                logging.info(
                    f'Shor succeed: randomly chosen a = {a}, who has common factor {gcd} with N classically')
                return gcd

            logging.info(f'round = {rd}')
            rd += 1
            # 4. Use quantum order-finding algorithm to find the order of a
            logging.info(f'Quantumly determine the order of the randomly chosen a = {a}')
            r = order_finding(a, N)
            if r == 0:
                logging.info(f'Shor failed: did not found the order of a = {a}')
            else:
                if r % 2 == 1:
                    logging.info(f'Shor failed: found odd order r = {r} of a = {a}')
                else:
                    h = pow(a, int(r / 2), N)
                    if h == N - 1:
                        logging.info(
                            f'Shor failed: found order r = {r} of a = {a} with negative square root')
                    else:
                        f1 = np.gcd(h - 1, N)
                        f2 = np.gcd(h + 1, N)
                        if f1 > 1 and f1 < N:
                            logging.info(
                                f'Shor succeed: found factor {f1}, with the help of a = {a}, r = {r}')
                            return f1
                        elif f2 > 1 and f2 < N:
                            logging.info(
                                f'Shor succeed: found factor {f2}, with the help of a = {a}, r = {r}')
                            return f2
                        else:
                            logging.info(f'Shor failed: can not find a factor with a = {a}')
