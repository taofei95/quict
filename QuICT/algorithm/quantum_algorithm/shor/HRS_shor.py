#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/29 15:28
# @Author  : Zhu Qinlin
# @File    : HRS_shor.py

'''
The (2n+2)-qubit circuit used in the Shor algorithm is designed by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in 
"Factoring using 2n+2 qubits with Toffoli based modular multiplication"
'''

from QuICT.core import *
from QuICT.qcda.synthesis.arithmetic.hrs import *
from QuICT.algorithm import Algorithm

import random
from math import pi
import numpy as np
from fractions import Fraction
from .utility import *

def order_finding(a,N):
    """
    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    #phase estimation procedure
    n = int(np.ceil(np.log2(N)))
    t = 2*n
    print('\torder_finding begin: circuit: L =',n,'t =',t)
    trickbit_store = [0]*t
    circuit = Circuit(2*n+2)
    x_reg = circuit([i for i in range(n)])
    #ancilla = circuit([i for i in range(n,2*n)])
    #indicator = circuit(2*n)
    trickbit = circuit(2*n+1)
    X | x_reg[n-1]
    for k in range(t):
        H | trickbit
        gate_pow = pow(a, 1<<(t-1-k), N)
        CHRSMulMod.execute(n, gate_pow, N) | circuit
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi /(1<<(k-i))) | trickbit
        H | trickbit
        
        Measure | trickbit
        circuit.exec()
        print('\tthe %dth trickbit measured to be %d'%(k,int(trickbit)))
        trickbit_store[k] = int(trickbit)
        if trickbit_store[k] == 1:
            X | trickbit
    Measure | x_reg
    trickbit_store.reverse()
    print('\tphi~ (approximately s/r) in binary form is',trickbit_store)

    #continued fraction procedure
    phi_ = sum([(trickbit_store[i]*1. / (1<<(i+1))) for i in range(t)])
    print('\tphi~ (approximately s/r) in decimal form is',phi_)
    if phi_ == 0.0:
        print('\torder_finding failed: phi~ = 0')
        return 0
    (num,den) = (Fraction(phi_).numerator,Fraction(phi_).denominator)
    CFE = continued_fraction_expansion(num,den)
    print('\tContinued fraction expansion of phi~ is',CFE)
    num1 = CFE[0]; den1 = 1; num2 = 1; den2 = 0
    print('\tthe 0th convergence is %d/%d'%(num1,den1))
    for k in range(1,len(CFE)):
        num = num1*CFE[k] + num2
        den = den1*CFE[k] + den2
        print('\tthe %dth convergence is %d/%d'%(k,num,den))
        if den >= N:
            break
        else:
            num2 = num1
            num1 = num
            den2 = den1
            den1 = den
    r = den1
    if pow(a,r,N) == 1:
        print('\torder_finding succeed: r = %d is the order of a = %d'%(r,a))
        return r
    else:
        print('\torder_finding failed: r = %d is not order of a = %d'%(r,a))
        return 0

def shor(N):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    """
    # 0. Check if input is prime (using MillerRabin in klog(N), k is the number of rounds to run MillerRabin)
    assert (not miller_rabin(N)), 'N is prime'

    # 1. If n is even, return the factor 2
    if N % 2 == 0:
        print('Shor succeed: N is even, found factor 2 classically')
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
            print('Shor succeed: N is exponential, found the only factor %d classically'%u1)
            return u1
        if pow(u2, b) == N:
            print('Shor succeed: N is exponential, found the only factor %d classically'%u2)
            return u2
    
    rd = 0
    max_rd = 15
    while rd<max_rd:
        # 3. Choose a random number a (1<a<N)
        a = random.randint(2,N-1)
        gcd = np.gcd(a,N)
        if gcd > 1:
            print('Shor succeed: randomly chosen a = %d, who has common factor %d with N classically'%(a,gcd))
            return gcd

        print('round =',rd)
        rd += 1
        # 4. Use quantum order-finding algorithm to find the order of a
        print('Quantumly determine the order of the randomly chosen a =',a)
        r = order_finding(a,N)
        if r == 0:
            print('Shor failed: did not found the order of a =',a)
        else:
            if r%2 == 1:
                print('Shor failed: found odd order r = %d of a = %d'%(r,a))
            else:
                h = pow(a,int(r/2),N)
                if h == N-1:
                    print('Shor failed: found order r = %d of a = %d with negative square root'%(r,a))
                else:
                    f1 = np.gcd(h-1,N); f2 = np.gcd(h+1,N)
                    if f1 > 1 and f1 < N:
                        print('Shor succeed: found factor %d, with the help of a = %d, r = %d'%(f1,a,r))
                        return f1
                    elif f2 > 1 and f2 < N:
                        print('Shor succeed: found factor %d, with the help of a = %d, r = %d'%(f2,a,r))
                        return f2
                    else:
                        print('Shor failed: can not find a factor with a = %d', a)

class HRSShorFactor(Algorithm):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    """
    @staticmethod
    def _run(N):
        return shor(N)
