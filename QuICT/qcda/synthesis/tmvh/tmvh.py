#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 10:55
# @Author  : Zhu Qinlin
# @File    : tmvh.py

from numpy import log2, floor, gcd, pi

from .._synthesis import Synthesis
from QuICT.core import Circuit, CX, CCX, CompositeGate, Swap, X, Ry, Rz, Measure
from QuICT.algorithm import Amplitude

def Set(qreg, N):
    """ Set the qreg as N, using X gates on specific qubits

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

def CV(control, target):
    Rz(pi/2)    | target
    Ry(pi/4)    | target
    CX          | (control,target)
    Ry(-pi/4)   | target
    CX          | (control,target)
    Rz(-pi/2)   | target

def CV_dagger(control,target):
    Rz(pi/2)    | target
    CX          | (control,target)
    Ry(pi/4)    | target
    CX          | (control,target)
    Ry(-pi/4)   | target
    Rz(pi/2)    | target

def PeresGate(a,b,c):
    """
    (a, b, c) -> (a, a xor b, a.b xor c)
    """
    '''
    CV_dagger(a,c)
    CV_dagger(b,c)
    CX | (a,b)
    CV(b,c)
    '''
    CCX | (a,b,c)
    CX  | (a,b)


def RippleCarryAdder1(a, b, overflow):
    '''
     store a + b in b

    (a,b,overflow) -> (a,b'=a+b,overflow'=overflow xor highest_carry)

    Args:
        a(Qureg): the qureg stores a, length is n
        b(Qureg): the qureg stores b, length is n
        overflow(Qureg): the ancillary qubits, length is 1

    reference: HIMANSHU THAPLIYAL and NAGARAJAN RANGANATHAN - Design of Efficient Reversible Logic Based Binary and BCD Adder Circuits
    '''

    n = len(a)

    #step 1
    for i in range(n-1):
        CX | (a[i],b[i])
    
    #step 2
    CX | (a[0],overflow)
    for i in range(n-2):
        CX | (a[i+1],a[i])
    
    #step 3
    for i in range(n-1):
        CCX | (a[n-1-i],b[n-1-i],a[n-2-i])
    
    #step 4
    PeresGate(a[0],b[0],overflow)
    for i in range(n-1):
        PeresGate(a[i+1],b[i+1],a[i])
    
    #step 5
    for i in range(n-2):
        CX | (a[n-2-i],a[n-3-i])
    
    #step 6
    for i in range(n-1):
        CX | (a[i],b[i])

def RippleCarryAdder(a, b):
    '''
     store a + b in b

    (a,b) -> (a,b'=a+b)

    Args:
        a(Qureg): the qureg stores a, length is n
        b(Qureg): the qureg stores b, length is n

    reference: HIMANSHU THAPLIYAL and NAGARAJAN RANGANATHAN - Design of Efficient Reversible Logic Based Binary and BCD Adder Circuits
    '''

    n = len(a)

    #step 1
    for i in range(n-1):
        CX | (a[i],b[i])
    
    #step 2
    for i in range(n-2):
        CX | (a[i+1],a[i])
    
    #step 3
    for i in range(n-1):
        CCX | (a[n-1-i],b[n-1-i],a[n-2-i])
    
    #step 4
    CX | (a[0],b[0])
    for i in range(n-1):
        PeresGate(a[i+1],b[i+1],a[i])
    
    #step 5
    for i in range(n-2):
        CX | (a[n-2-i],a[n-3-i])
    
    #step 6
    for i in range(n-1):
        CX | (a[i],b[i])

def Subtraction(a,b):
    """
    a-b
    """
    X | a
    RippleCarryAdder(a, b)
    X | a

def CtrlAdd(ctrl,a,b):
    pass

def RestoringDivision(a,b,r):
    """
    Divided: a
    Divisor: b
    """
    n = len(a)

    for i in range(n):
        y = r[i+1:n] + a[0:i+1]
        #Iteration(y,b,r[i])
        Subtraction(y,b)
        CX | (r[i+1],r[i])
        CtrlAdd(r[i],y,b)
        X | r[i]

def VBEDecomposition(m, a, N):
    """ give parameters to the VBE
    Args:
        m(int): number of qubits of x
        a(int): a
        N(int): N
    Returns:
        CompositeGate: the model filled by parameters.
    """
    if N <= 2:
        raise Exception("modulus should be great than 2")
    if gcd(a, N) != 1:
        raise Exception("a and N should be co-prime")
    n = int(floor(log2(N))) + 1

    circuit = Circuit(m + 5 * n + 2)
    qubit_x = circuit([i for i in range(m)])
    qubit_r = circuit([i for i in range(m, m + n)])
    qubit_a = circuit([i for i in range(m + n, m + 2 * n)])
    qubit_b = circuit([i for i in range(m + 2 * n, m + 3 * n)])
    qubit_c = circuit([i for i in range(m + 3 * n, m + 4 * n)])

    overflow = circuit(m + 4 * n)
    qubit_N = circuit([i for i in range(m + 4 * n + 1, m + 5 * n + 1)])
    t = circuit(m + 5 * n + 1)
    X | qubit_r[n - 1]
    RestoringDivision(qubit_a,qubit_b,qubit_r)
    return CompositeGate(circuit.gates)

VBE = Synthesis(VBEDecomposition)


(astr,bstr) = input("input a, b: ").split()

a = int(astr)
b = int(bstr)

n = max(len(bin(a))-2,len(bin(b))-2)

circuit = Circuit(2*n+1)
a_q = circuit([i for i in range(n)])
b_q = circuit([i for i in range(n,2*n)])
o_q = circuit(2*n)
Set(a_q,a)
Set(b_q,b)
RippleCarryAdder(a_q,b_q)
Measure | circuit
circuit.exec()

print(int(a_q),int(b_q),int(o_q))