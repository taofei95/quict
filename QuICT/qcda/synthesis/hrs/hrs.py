#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 13:12
# @Author  : Li Haomin
# @File    : hrs.py

from numpy import log2, floor, gcd

from .._synthesis import Synthesis
from QuICT.core import Circuit, CX, CCX, Swap, X

def EX_GCD(a, b, arr):
    """ 
    Implementation of Extended Euclidean algorithm

    Args:
        a(int): the parameter a
        b(int): the parameter b
        arr(list): store the solution of ax + by = gcd(a, b) in arr, length is 2

    """

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
    """ 
    Inversion of a in (mod N)

    Args:
        a(int): the parameter a
        n(int): the parameter n

    """
    arr = [0, 1]
    EX_GCD(a, n, arr)
    return (arr[0] % n + n) % n

def int2bitwise(c, n):
    """ 
    Transform an integer c to binary n-length bitwise string.
    
    Args:
        c(int) the parameter c
        n(int) the parameter n
    """
    c_bitwise = bin(c)[2:]
    if len(c_bitwise) > n:
        c_bitwise = c_bitwise[-n:]
        #print('c exceeds the length of a, thus is truncated')
    else:
        c_bitwise = '0'*(n-len(c_bitwise))+c_bitwise
    return c_bitwise

def fast_power(a, b, N):
    """ 
    Implementation of Fase Power algorithm, calculate a^b mod N

    Args:
        q(int): the parameter a
        b(int): the parameter b
        N(int): the parameter N

    """    
    x = 1
    now_a = a
    while b > 0:
        if b % 2 == 1:
            x = x * now_a % N
        now_a = now_a * now_a % N
        b >>= 1
    return x

def Split_Invert(n,d,CFE):
    """ 
    Recursive expansion part of CFE

    Args:
        n(int): numerator
        d(int): denominator
        CFE(list): store the result of expansion

    """

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
        d: denominator

    """
    CFE = []
    Split_Invert(n,d,CFE)
    return CFE

def Set(qreg, N):
    """ 
    Set the qreg as N, using X gates on specific qubits.
    
    Args:
        qreg(Qureg): the qureg to be set
        N(int): the parameter N    

    """
    string = bin(N)[2:]
    n = len(qreg)
    m = len(string)
    if m > n:
        print('When set qureg as N=%d, N exceeds the length of qureg n=%d, thus is truncated' % (N, n))
    
    for i in range(min(n, m)):
        if string[m - 1 - i] == '1':
            X | qreg[n - 1 - i]

def CCarry(control,a,c_bitwise,g_aug,overflow):
    """
    1-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in Carry() to CCX.

    Args:
        control: 1 qubit.
        a: n qubits.
        g_aug: n-1 qubits(more bits are OK).
        overflow: 1 qubit.
        c_bitwise: n bits.
    """
    n = len(a)
    g = g_aug[0:n-1]
    #n==1, no borrowed bits g
    if n==1:
        if c_bitwise[0]=='1':
            CCX | (control,a[0],overflow)
        return
    #n>=2
    CCX | (control,g[0],overflow)
    
    for i in range(n-2):
        if c_bitwise[i] == '1':
            CX | (a[i],g[i])
            X  | a[i]
        CCX | (g[i+1],a[i],g[i])
    if c_bitwise[n-2] == '1':
        CX | (a[n-2],g[n-2])
        X  | a[n-2]
    if c_bitwise[n-1] == '1':
        CCX | (a[n-1],a[n-2],g[n-2])
    for i in range(n-2):
        CCX | (g[n-2-i],a[n-3-i],g[n-3-i])
    
    CCX | (control,g[0],overflow)
    
    #uncomputation
    for i in range(n-2):
        CCX | (g[i+1],a[i],g[i])
    if c_bitwise[n-1] == '1':
        CCX | (a[n-1],a[n-2],g[n-2])
    if c_bitwise[n-2] == '1':
        X  | a[n-2]
        CX | (a[n-2],g[n-2])
    for i in range(n-2):
        CCX | (g[n-2-i],a[n-3-i],g[n-3-i])
        if c_bitwise[n-3-i] == '1':
            X  | a[n-3-i]
            CX | (a[n-3-i],g[n-3-i])


def CCCarry(control1,control2,a,c_bitwise,g_aug,overflow):
    """
    2-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in Carry() to CCCX.

    Args:
        control1: 1 qubit.
        control2: 1 qubit.
        a: n qubits .
        g_aug: n-1 qubits(more bits are OK).
        overflow: 1 qubit.
        c_bitwise:n bits.
    """
    n = len(a)
    #n==1, no borrowed bits g
    if n==1:
        if c_bitwise[0]=='1':
            #CCCX | (c1,c2,a[0],overflow) with g_aug[0] as ancilla
            CCX | (a[0],g_aug[0],overflow)
            CCX | (control1,control2,g_aug[0])
            CCX | (a[0],g_aug[0],overflow)
            CCX | (control1,control2,g_aug[0])
        return
    #n>=2
    g = g_aug[0:n-1]
    #CCCX | (c1,c2,g[0],overflow) with a[0] as ancilla
    CCX | (g[0],a[0],overflow)
    CCX | (control1,control2,a[0])
    CCX | (g[0],a[0],overflow)
    CCX | (control1,control2,a[0])
    
    for i in range(n-2):
        if c_bitwise[i] == '1':
            CX | (a[i],g[i])
            X  | a[i]
        CCX | (g[i+1],a[i],g[i])
    if c_bitwise[n-2] == '1':
        CX | (a[n-2],g[n-2])
        X  | a[n-2]
    if c_bitwise[n-1] == '1':
        CCX | (a[n-1],a[n-2],g[n-2])
    for i in range(n-2):
        CCX | (g[n-2-i],a[n-3-i],g[n-3-i])
    
    #CCCX | (c1,c2,g[0],overflow) with a[0] as ancilla
    CCX | (g[0],a[0],overflow)
    CCX | (control1,control2,a[0])
    CCX | (g[0],a[0],overflow)
    CCX | (control1,control2,a[0])
    
    #uncomputation
    for i in range(n-2):
        CCX | (g[i+1],a[i],g[i])
    if c_bitwise[n-1] == '1':
        CCX | (a[n-1],a[n-2],g[n-2])
    if c_bitwise[n-2] == '1':
        X  | a[n-2]
        CX | (a[n-2],g[n-2])
    for i in range(n-2):
        CCX | (g[n-2-i],a[n-3-i],g[n-3-i])
        if c_bitwise[n-3-i] == '1':
            X  | a[n-3-i]
            CX | (a[n-3-i],g[n-3-i])

def SubWidget(v,g):
    """
        Subwidget used in Incrementer().

        Args:
            v: n qubits.
            g: n qubits(more qubits are OK).
    """
    n = len(v)
    if len(g)<n:
        print('When do Sub_Widget, no edequate ancilla qubit')
    
    for i in range(n-1):
        CX  | (g[n-1-i],v[n-1-i])
        CX  | (g[n-2-i],g[n-1-i])
        CCX | (g[n-1-i],v[n-1-i],g[n-2-i])
    CX | (g[0],v[0])
    for i in range(n-1):
        CCX | (g[i+1],v[i+1],g[i])
        CX  | (g[i],g[i+1])
        CX  | (g[i],v[i+1])


def Incrementer(v,g):
    """
    Incremente v by 1, with borrowed qubits g.

    Args:
        v: n qubits.
        g: n qubits(more qubits are OK).
    """
    n = len(v)
    if len(g)<n:
        print('When do Increment, no edequate borrowed qubit')
    
    for i in range(n):
        CX | (g[n-1],v[i])
    for i in range(n-1):
        X | g[i]
    X | v[0]
    SubWidget(v,g)
    for i in range(n-1):
        X | g[i]
    SubWidget(v,g)
    for i in range(n):
        CX | (g[n-1],v[i])


class HRSModel(Synthesis):
    """ 
    parent class

    """
    def __call__(self, *pargs):
        """ 
        Calling this empty class makes no effect
        """
        raise Exception('Calling this empty class makes no effect')

    def build_gate(self):
        """ 
        Empty class builds no gate
        """
        raise Exception('Empty class builds no gate')

HRS = HRSModel()

class HRSIncrementerModel(HRSModel):
    def __call__(self,n):
        self.pargs = [n]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the VBE circuit
        """
        n = self.pargs[0]

        circuit = Circuit(2 * n)
        qubit_a = circuit([i for i in range(n)])
        qubit_g = circuit([i for i in range(n, 2*n)])
        Incrementer(qubit_a, qubit_g)
        return circuit    

HRSIncrementer = HRSIncrementerModel()

