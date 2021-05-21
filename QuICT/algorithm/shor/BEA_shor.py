#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/29 15:28
# @Author  : Zhu Qinlin
# @File    : HRS_shor.py

'The (2n+2)-qubit circuit used in the Shor algorithm is designed by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication'

from QuICT.core import *
import sys
from .._algorithm import Algorithm

import random
from math import log, ceil, floor, gcd, pi
import numpy as np
from fractions import Fraction
# from QuICT.algorithm import Amplitude
import time

# from QuICT.core import *
from QuICT.qcda.synthesis.arithmetic.bea import *

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

#transform an integer to n-length bitwise string
def int2bitwise(c,n):
    """
    Transform an integer c to binary n-length bitwise string.
    """
    c_bitwise = bin(c)[2:]
    if len(c_bitwise) > n:
        c_bitwise = c_bitwise[-n:]
        #print('c exceeds the length of a, thus is truncated')
    else:
        c_bitwise = '0'*(n-len(c_bitwise))+c_bitwise
    return c_bitwise

def fast_power(a, b, N):
    x = 1
    now_a = a
    while b > 0:
        if b % 2 == 1:
            x = x * now_a % N
        now_a = now_a * now_a % N
        b >>= 1
    return x

def Split_Invert(n,d,CFE):
    CFE.append(n//d)
    n = n%d
    if n == 1:
        CFE.append(d)
        return
    Split_Invert(d,n,CFE)

def Continued_Fraction_Expansion(n,d):
    """
    Calculate the continued fraction expansion of a rational number n/d.

    Args:
        n: numerator.
        d: denominator.
    """
    CFE = []
    Split_Invert(n,d,CFE)
    return CFE

def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits.
    """
    str = bin(N)[2:]
    n = len(qreg); m = len(str)
    if m > n:
        print('Warning: When set qureg as N=%d, N exceeds the length of qureg n=%d, thus is truncated'%(N,n))
    
    for i in range(min(n,m)):
        if str[m-1-i] == '1':
            X | qreg[n-1-i]

def Order_Finding(a,N):
    """
    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    #phase estimation procedure
    n = int(np.ceil(np.log2(N)))
    t = 2*n
    print('\tOrder_Finding begin: circuit: L =',n,'t =',t)
    trickbit_store = [0]*t
    circuit = Circuit(2*n+3)
    x_reg = circuit([i for i in range(n+1,2*n+1)])
    ancilla = circuit([i for i in range(n+1)])
    trickbit = circuit(2 * n + 1)
    qreg_low= circuit(2 * n + 2)
    X | x_reg[n-1]
    for k in range(t):
        H | trickbit
        gate_pow = pow(a, 1<<(t-1-k), N)
        BEACUa(n, gate_pow, N) | circuit
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
        print('\tOrder_Finding failed: phi~ = 0')
        return 0
    (num,den) = (Fraction(phi_).numerator,Fraction(phi_).denominator)
    CFE = Continued_Fraction_Expansion(num,den)
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
        print('\tOrder_Finding succeed: r = %d is the order of a = %d'%(r,a))
        return r
    else:
        print('\tOrder_Finding failed: r = %d is not order of a = %d'%(r,a))
        return 0

def MillerRabin(num):
    Test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if num == 1:
        return False
    t = num - 1
    k = 0
    while (t & 1) == 0:
        k += 1
        t >>= 1
    for test_num in Test:
        # test_num should be generated randomly
        if num == test_num:
            return True
        a = fast_power(test_num, t, num)
        nxt = a
        for _ in range(k):
            nxt = (a * a) % num
            if nxt == 1 and a != 1 and a != num - 1:
                return 0
            a = nxt
        if a != 1:
            return False
    return True

def Shor(N):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    """
    # check if input is prime (using MillerRabin in klog(N), k is the number of rounds to run MillerRabin)
    assert (not MillerRabin(N)), 'N is prime'

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
        r = Order_Finding(a,N)
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

class BEAShorFactor(Algorithm):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    """
    @staticmethod
    def _run(N):
        return Shor(N)

if __name__ == "__main__":
    time_start = time.time_ns()
    BEAShorFactor.run(21)
    time_end = time.time_ns()
    print(time_end - time_start)