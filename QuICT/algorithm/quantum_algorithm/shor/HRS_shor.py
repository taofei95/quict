#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/29 15:28
# @Author  : Zhu Qinlin
# @File    : HRS_shor.py

'The (2n+2)-qubit circuit used in the Shor algorithm is designed by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication'

from QuICT.core import *
from QuICT.algorithm import Algorithm

import random
from math import log, ceil, floor, gcd, pi
import numpy as np
from fractions import Fraction
import time
from .utility import *

def split_invert(n,d,CFE):
    CFE.append(n//d)
    n = n%d
    if n == 1:
        CFE.append(d)
        return
    split_invert(d,n,CFE)

def continued_fraction_expansion(n,d):
    """
    Calculate the continued fraction expansion of a rational number n/d.

    Args:
        n: numerator.
        d: denominator.
    """
    CFE = []
    split_invert(n,d,CFE)
    return CFE

def c_carry(control,a,c_bitwise,g_aug,overflow):
    """
    1-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in Carry() to CCX.

    Args:
        control: 1 qubit.
        a: n qubits.
        g_aug: n-1 qubits(more bits are OK).
        overflow: 1 qubit.
        c_bitwise:n bits.
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


def cc_carry(control1,control2,a,c_bitwise,g_aug,overflow):
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


def sub_widget(v,g):
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


def incrementer(v,g):
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
    sub_widget(v,g)
    for i in range(n-1):
        X | g[i]
    sub_widget(v,g)
    for i in range(n):
        CX | (g[n-1],v[i])


def c_incrementer(control, v, g_aug):
    """
    1-controlled incremente v by 1, with borrowed qubits g.

    Constructed by attaching the control qubit to the little-end of v, and apply an (n+1)-bit Incrementer() to it.

    Args:
        control: 1 qubit.
        v: n qubits.
        g: n+1 qubits(more qubits are OK).
    """
    n = len(v)
    m = len(g_aug)
    if m<n+1 :
        print("no edequate ancilla bits")
    g = g_aug[0:n+1]
    vc = v + control
    incrementer(vc,g)
    X | vc[n]

def c_adder_rec(control,x,c_bitwise,ancilla,ancilla_g):
    """
    The recursively applied partial-circuit in CAdder().
    
    Constructed by changing the Carry() in Adder_rec() to CCarry().

    Args:
        control: 1 qubit.
        x: n qubits.
        ancilla: 1 qubit.
        ancilla_g: 1 qubit, might be used as borrowed qubit in CIncrementer when x_H and x_L are of the same length.
        c_bitwise: n bits.
    """
    n = len(x)
    if n==1:
        return
    mid = n//2
    x_H = x[0:mid]
    x_L = x[mid:n]
    c_H = c_bitwise[0:mid]
    c_L = c_bitwise[mid:n]
    g = x_L + ancilla_g
    c_incrementer(ancilla,x_H,g)
    for i in range(mid):
        CX | (ancilla, x_H[i])
    c_carry(control,x_L,c_L,x_H,ancilla)
    c_incrementer(ancilla,x_H,g)
    c_carry(control,x_L,c_L,x_H,ancilla)
    for i in range(mid):
        CX | (ancilla, x_H[i])
    c_adder_rec(control,x_L,c_L,ancilla,ancilla_g)
    c_adder_rec(control,x_H,c_H,ancilla,ancilla_g)


def c_adder(control,x,c,ancilla,ancilla_g):
    """
    Compute x(quantum) + c(classical) with borrowed qubits, 1-controlled.

    Args:
        control: 1 qubit.
        x: n qubits.
        ancilla: 1 qubit, borrowed ancilla.
        ancilla_g: 1 qubit, borrowed ancilla.
        c: integer.
    """
    n = len(x)
    c_bitwise = int2bitwise(c,n)
    c_adder_rec(control,x,c_bitwise,ancilla,ancilla_g)
    #print(Amplitude.run(circuit))
    for i in range(n):
        if c_bitwise[i]=='1':
            CX | (control,x[i])


def c_sub(control,x,c,ancilla,ancilla_g):
    """
    Compute x(quantum) + c(classical) with borrowed qubits, 1-controlled.

    Constructed on the basis of CAdder() with complement technique.

    Args:
        control: 1 qubit.
        x: n qubits.
        ancilla: 1 qubit, borrowed ancilla.
        ancilla_g: 1 qubit, borrowed ancilla.
        c: integer.
    """
    n = len(x)
    c_complement = 2**n-c
    cc_bitwise = int2bitwise(c_complement,n)
    c_adder_rec(control,x,cc_bitwise,ancilla,ancilla_g)
    for i in range(n):
        if cc_bitwise[i]=='1':
            CX | (control,x[i])

#controlled compare b and c. indicator toggles if c > b, not if c <= b
def cc_compare(control1,control2,b,c,g_aug,indicator):
    """
    Compare b and c with borrowed qubits g_aug. The Indicator toggles if c > b, not if c <= b, 2controlled.

    Constructed on the basis of CCCarry().

    Args:
        control1: 1 qubit.
        control2: 1 qubit.
        b: n qubits.
        g_aug: n-1 qubits(more qubits are OK).
        indicator: 1 qubit.
        c: integer with less-than-n length.
    """
    n = len(b)
    m = len(g_aug)
    if m < n-1:
        print('No edequate ancilla bits when compare\n')
        return
    c_bitwise = int2bitwise(c,n)
    X | b
    c_carry(control1,control2,b,c_bitwise,g_aug,indicator)
    X | b

#b: n bit, g: n-1 bit, indicator: 1 bit
def cc_adder_mod(control1,control2,b,a,N,g,indicator):
    """
    Compute b(quantum) + a(classical) mod N(classical), with borrowed qubits g and ancilla qubit indicator, 2-controlled.

    Args：
        control1: 1 qubit.
        control2：1 qubit.
        b: n qubits.
        g: n-1 borrowed qubits(more qubits are OK).
        indicator: 1 ancilla qubit.
        a: integer less than N.
        N: integer.
    """
    cc_compare(control1,control2,b,N-a,g,indicator)
    c_adder(indicator,b,a,g[0:1],g[1:2])
    CCX | (control1,control2,indicator)
    c_sub(indicator,b,N-a,g[0:1],g[1:2])
    CCX | (control1,control2,indicator)
    cc_compare(control1,control2,b,a,g,indicator)
    CCX | (control1,control2,indicator)


def cc_adder_mod_reverse(control1,control2,b,a,N,g,indicator):
    """
    The reversed circuit of CCAdder_Mod()
    """
    cc_adder_mod(control1,control2,b,N-a,N,g,indicator)


#x: n bits, b: n bits
def c_mul_mod_raw(control,x,a,b,N,indicator):
    """
    Compute b(quantum) + x(quantum) * a(classical) mod N(classical), with target qubits b and ancilla qubit indicator, 1-controlled.

    Args:
        control: 1 qubit.
        x: n qubits.
        b: n qubits, target.
        indicator: 1 ancilla qubit.
        a: integer.
        N: integer.
    """
    n = len(x)
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a*2 %N
    for i in range(n):
        #borrow all the n-1 unused qubits in x
        g = x[:n-i-1]+x[n-i:]
        cc_adder_mod(control,x[n-1-i],b,a_list[i],N,g,indicator)


def c_mul_mod_raw_reverse(control,x,a,b,N,indicator):
    n = len(x)
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a*2 %N
    for i in range(n):
        g = x[:i]+x[i+1:]
        cc_adder_mod(control,x[i],b,N-a_list[n-i-1],N,g,indicator)


#x: n bits, ancilla: n bits, indicator: 1 bit
def c_mul_mod(control,x,a,ancilla,N,indicator):
    """
    Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits, 1-controlled.

    Args:
        control: 1 qubit.
        x: n qubits.
        ancilla: n qubits.
        indicator: 1 qubit.
        a: integer.
        N: integer.
    """
    n = len(x)
    a_r = ModReverse(a,N)
    c_mul_mod_raw(control,x,a,ancilla,N,indicator)
    #CSwap
    for i in range(n):
        CSwap(control,x[i],ancilla[i])
    c_mul_mod_raw_reverse(control,x,a_r,ancilla,N,indicator)


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
    ancilla = circuit([i for i in range(n,2*n)])
    indicator = circuit(2*n)
    trickbit = circuit(2*n+1)
    X | x_reg[n-1]
    for k in range(t):
        H | trickbit
        gate_pow = pow(a, 1<<(t-1-k), N)
        c_mul_mod(trickbit,x_reg,gate_pow,ancilla,N,indicator)
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


def Shor(N):
    """
    Shor algorithm by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in "Factoring using 2n+2 qubits with Toffoli based modular multiplication"
    """

    # 1. If n is even, return the factor 2
    if N < 4:
        print('Shor ignored: N is too small(<4) to use HRS shor circuit')
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
    while True:
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
                    if f1 > 1:
                        print('Shor succeed: found factor %d, with the help of a = %d, r = %d'%(f1,a,r))
                        return f1
                    elif f2 > 1:
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
        return Shor(N)