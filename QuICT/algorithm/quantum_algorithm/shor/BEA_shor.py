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


def order_finding(a:int, N: int, demo = None, eps: float = 1/10,):
    """
    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1/(2*eps))))
    msg = f'\torder_finding begin: circuit: L = {n} t = {t}'
    if demo == 'demo': print(msg)
    else: logging.info(msg)
    trickbit_store = [0] * t
    circuit = Circuit(2 * n + 3)
    # ancilla = circuit([i for i in range(n+1)])
    x_reg = circuit([i for i in range(n + 1, 2 * n + 1)])
    trickbit = circuit(2 * n + 1)
    qreg_low= circuit(2 * n + 2)
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
        # Measure | qreg_low
        circuit.exec()
        # assert int(qreg_low)==0
        msg = f'\tthe {k}th trickbit measured to be {int(trickbit)}'
        if demo == 'demo': print(msg)
        else: logging.info(msg)
        trickbit_store[k] = int(trickbit)
        if trickbit_store[k] == 1:
            X | trickbit
    Measure | x_reg
    trickbit_store.reverse()
    msg = f'\tphi~ (approximately s/r) in binary form is {trickbit_store}'
    if demo == 'demo': print(msg)
    else: logging.info(msg)
    # continued fraction procedure
    phi_ = sum([(trickbit_store[i] * 1. / (1 << (i + 1))) for i in range(t)])
    msg = f'\tphi~ (approximately s/r) in decimal form is {phi_}'
    if demo == 'demo': print(msg)
    else: logging.info(msg)
    if phi_ == 0.0:
        msg = '\torder_finding failed: phi~ = 0'
        if demo == 'demo': print(msg)
        else: logging.info(msg)
        return 0
    (num, den) = (Fraction(phi_).numerator, Fraction(phi_).denominator)
    CFE = continued_fraction_expansion(num, den)
    msg = f'\tContinued fraction expansion of phi~ is {CFE}'
    if demo == 'demo': print(msg)
    else: logging.info(msg)
    num1 = CFE[0]
    den1 = 1
    num2 = 1
    den2 = 0
    msg = f'\tthe 0th convergence is {num1}/{den1}'
    if demo == 'demo': print(msg)
    else: logging.info(msg)
    for k in range(1, len(CFE)):
        num = num1 * CFE[k] + num2
        den = den1 * CFE[k] + den2
        msg = f'\tthe {k}th convergence is {num}/{den}'
        if demo == 'demo': print(msg)
        else: logging.info(msg)
        if den >= N:
            break
        else:
            num2 = num1
            num1 = num
            den2 = den1
            den1 = den
    r = den1
    if pow(a, r, N) == 1:
        msg = f'\torder_finding succeed: r = {r} is the order of a = {a}'
        if demo == 'demo': print(msg)
        else: logging.info(msg)
        return r
    else:
        msg = f'\torder_finding failed: r = {r} is not the order of a = {a}'
        if demo == 'demo': print(msg)
        else: logging.info(msg)
        return 0

class BEA_order_finding(Algorithm):
    '''
    The (2n+2)-qubit circuit used in the order_finding algorithm is designed by \
    THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE \
    in "Factoring using 2n+2 qubits with Toffoli based modular multiplication\
    '''
    @staticmethod
    def run(a: int, N: int, demo:str = None):
        return order_finding(a, N, demo)

class BEAShorFactor(Algorithm):
    '''
    The (2n+2)-qubit circuit used in the Shor algorithm is designed by \
    THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE \
    in "Factoring using 2n+2 qubits with Toffoli based modular multiplication\
    '''
    @staticmethod
    def run(N: int, max_rd: int,  demo:str = None, eps: float = 1/10,):
        # check if input is prime (using MillerRabin in klog(N), k is the number of rounds to run MillerRabin)
        if (miller_rabin(N)):
            msg = f'N does not pass miller rabin test, may be a prime number'
            if demo == 'demo': print(msg)
            else: logging.info(msg)
            return 0

        # 1. If n is even, return the factor 2
        if N % 2 == 0:
            msg = 'Shor succeed: N is even, found factor 2 classically'
            if demo == 'demo': print(msg)
            else: logging.info(msg)
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
                msg = f'Shor succeed: N is exponential, found the only factor {u1} classically'
                if demo == 'demo': print(msg)
                else: logging.info(msg)
                return u1
            if pow(u2, b) == N:
                msg = f'Shor succeed: N is exponential, found the only factor {u2} classically'
                if demo == 'demo': print(msg)
                else: logging.info(msg)
                return u2

        rd = 0
        while rd < max_rd:
            # 3. Choose a random number a (1<a<N)
            a = random.randint(2, N - 1)
            gcd = np.gcd(a, N)
            if gcd > 1:
                msg = f'Shor succeed: randomly chosen a = {a}, who has common factor {gcd} with N classically'
                if demo == 'demo': print(msg)
                else: logging.info(msg)
                return gcd

            msg = f'round = {rd}'
            if demo == 'demo': print(msg)
            else: logging.info(msg)
            rd += 1
            # 4. Use quantum order-finding algorithm to find the order of a
            msg = f'Quantumly determine the order of the randomly chosen a = {a}'
            if demo == 'demo': print(msg)
            else: logging.info(msg)
            r = order_finding(a, N, demo, eps)
            if r == 0:
                msg = f'Shor failed: did not find the order of a = {a}'
                if demo == 'demo': print(msg)
                else: logging.info(msg)
            else:
                if r % 2 == 1:
                    msg = f'Shor failed: found odd order r = {r} of a = {a}'
                    if demo == 'demo': print(msg)
                    else: logging.info(msg)
                else:
                    h = pow(a, int(r / 2), N)
                    if h == N - 1:
                        msg = f'Shor failed: found order r = {r} of a = {a} with negative square root'
                        if demo == 'demo': print(msg)
                        else: logging.info(msg)
                    else:
                        f1 = np.gcd(h - 1, N)
                        f2 = np.gcd(h + 1, N)
                        if f1 > 1 and f1 < N:
                            msg = f'Shor succeed: found factor {f1}, with the help of a = {a}, r = {r}'
                            if demo == 'demo': print(msg)
                            else: logging.info(msg)
                            return f1
                        elif f2 > 1 and f2 < N:
                            msg = f'Shor succeed: found factor {f2}, with the help of a = {a}, r = {r}'
                            if demo == 'demo': print(msg)
                            else: logging.info(msg)
                            return f2
                        else:
                            msg = f'Shor failed: can not find a factor with a = {a}'
                            if demo == 'demo': print(msg)
                            else: logging.info(msg)
