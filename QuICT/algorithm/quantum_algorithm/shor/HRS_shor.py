#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/29 15:28
# @Author  : Zhu Qinlin
# @File    : HRS_shor.py

'''
The (2n+2)-qubit circuit used in the Shor algorithm is designed \
by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in \
"Factoring using 2n+2 qubits with Toffoli based modular multiplication"
'''

import random
import logging
from math import pi
import numpy as np
from fractions import Fraction

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.hrs import *
from QuICT.algorithm import Algorithm
from .utility import *

from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation import Simulator


def order_finding(a:int, N: int, demo = None, eps: float = 1/10, simulator: Simulator = UnitarySimulator()):
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

    circuit = Circuit(2 * n + 2)
    x_reg = [i for i in range(n)]
    # ancilla = circuit([i for i in range(n,2*n)])
    # indicator = circuit(2*n)
    trickbit = [2 * n + 1]
    X | circuit(x_reg[n - 1])
    for k in range(t):
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        CHRSMulMod.execute(n, gate_pow, N) | circuit
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | circuit(trickbit)
        H | circuit(trickbit)

        Measure | circuit(trickbit)
        simulator.run(circuit)
        msg = f'\tthe {k}th trickbit measured to be {int(circuit[trickbit])}'
        if demo == 'demo': print(msg)
        else: logging.info(msg)
        trickbit_store[k] = int(circuit[trickbit])
        if trickbit_store[k] == 1:
            X | circuit(trickbit)
    for idx in x_reg: Measure | circuit(idx)
    
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
    return r

class HRS_order_finding_twice(Algorithm):
    '''
    Run order_finding twice and take the lcm of the two result 
    to guaruntee a higher possibility to get the correct order,
    as suggested in QCQI 5.3.1
    '''
    @staticmethod
    def run(a: int, N: int, demo:str = None, eps: float = 1/10, simulator: Simulator = UnitarySimulator()):
        r1 = order_finding(a, N, demo, eps, simulator)
        r2 = order_finding(a, N, demo, eps, simulator)
        flag1 = (pow(a, r1, N) == 1 and r1!= 0)
        flag2 = (pow(a, r2, N) == 1 and r2!= 0)
        if flag1 and flag2:
            r = min(r1, r2)
        elif not flag1 and not flag2:
            r = int(np.lcm(r1,r2))
        else:
            r = int(flag1)*r1 + int(flag2)*r2
            
        if (pow(a,r,N)==1 and r!=0): 
            msg = f'\torder_finding found candidate order: r = {r} of a = {a}'
        else:  
            r = 0
            msg = f'\torder_finding failed'
        if demo == 'demo': print(msg)
        else: logging.info(msg)

        return r

class HRSShorFactor(Algorithm):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE \
    in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    """
    @staticmethod
    def run(N: int, max_rd: int,  demo:str = None, eps: float = 1/10, simulator: Simulator = UnitarySimulator()):
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
            r = HRS_order_finding_twice.run(a, N, demo, eps, simulator)
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