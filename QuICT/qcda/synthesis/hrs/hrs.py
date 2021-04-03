#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 13:12
# @Author  : Li Haomin
# @File    : hrs.py

from QuICT.core import Circuit, CX, CCX, Swap, X

def EX_GCD(a, b, arr):
    """ Implementation of Extended Euclidean algorithm

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
    """ Inversion of a in (mod N)

    Args:
        a(int): the parameter a
        n(int): the parameter n

    """
    arr = [0, 1]
    EX_GCD(a, n, arr)
    return (arr[0] % n + n) % n

def int2bitwise(c, n):
    """ Transform an integer c to binary n-length bitwise string.
    
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
    """ Implementation of Fase Power algorithm, calculate a^b mod N

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
    """ Recursive expansion part of CFE

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
    """ Calculate the continued fraction expansion of a rational number n/d.

    Args:
        n: numerator.
        d: denominator

    """
    CFE = []
    Split_Invert(n,d,CFE)
    return CFE

def Set(qreg, N):
    """ Set the qreg as N, using X gates on specific qubits.
    
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
        if str[m - 1 - i] == '1':
            X | qreg[n - 1 - i]


