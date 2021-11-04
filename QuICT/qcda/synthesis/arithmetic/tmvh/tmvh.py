#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 10:55
# @Author  : Zhu Qinlin
# @File    : tmvh.py

from QuICT.core import Circuit, CX, CCX, CompositeGate, X
from ..._synthesis import Synthesis

def peres_gate(a, b, c):
    """
    (a, b, c) -> (a, a xor b, a.b xor c)

    Args:
        a(Qubit): qubit
        b(Qubit): qubit
        c(Qubit): qubit
    """
    CCX | (a, b, c)
    CX | (a, b)


def adder_overflow(a, b, overflow):
    """
     store a + b in b

    (a,b,overflow) -> (a,b'=a+b,overflow'=overflow xor highest_carry)

    Args:
        a(Qureg): the qureg stores a, length is n
        b(Qureg): the qureg stores b, length is n
        overflow(Qureg): the ancillary qubits, length is 1

    reference: HIMANSHU THAPLIYAL and NAGARAJAN RANGANATHAN -
    Design of Efficient Reversible Logic Based Binary and BCD Adder Circuits
    https://arxiv.org/abs/1712.02630v1
    """

    n = len(a)

    if n == 1:
        peres_gate(a, b, overflow)
        return

    # step 1
    for i in range(n - 1):
        CX | (a[i], b[i])

    # step 2
    CX | (a[0], overflow)
    for i in range(n - 2):
        CX | (a[i + 1], a[i])

    # step 3
    for i in range(n - 1):
        CCX | (a[n - 1 - i], b[n - 1 - i], a[n - 2 - i])

    # step 4
    peres_gate(a[0], b[0], overflow)
    for i in range(n - 1):
        peres_gate(a[i + 1], b[i + 1], a[i])

    # step 5
    for i in range(n - 2):
        CX | (a[n - 2 - i], a[n - 3 - i])

    # step 6
    for i in range(n - 1):
        CX | (a[i], b[i])


def adder(a, b):
    """
    (a,b) -> (a,b'=a+b)

    Args:
        a(Qureg): the qureg stores a, length is n
        b(Qureg): the qureg stores b, length is n

    reference: HIMANSHU THAPLIYAL and NAGARAJAN RANGANATHAN -
    Design of Efficient Reversible Logic Based Binary and BCD Adder Circuits
    https://arxiv.org/abs/1712.02630v1
    """

    n = len(a)

    # step 1
    for i in range(n - 1):
        CX | (a[i], b[i])

    # step 2
    for i in range(n - 2):
        CX | (a[i + 1], a[i])

    # step 3
    for i in range(n - 1):
        CCX | (a[n - 1 - i], b[n - 1 - i], a[n - 2 - i])

    # step 4
    CX | (a[0], b[0])
    for i in range(n - 1):
        peres_gate(a[i + 1], b[i + 1], a[i])

    # step 5
    for i in range(n - 2):
        CX | (a[n - 2 - i], a[n - 3 - i])

    # step 6
    for i in range(n - 1):
        CX | (a[i], b[i])


def subtraction(a, b):
    """
    (a,b) -> (a,b-a)
    """
    X | b
    adder(a, b)
    X | b


def subtraction_overflow(a, b, overflow):
    """
    (a,b) -> (a,b-a)
    """
    X | b
    X | overflow
    adder_overflow(a, b, overflow)
    X | b
    X | overflow

def ctrl_add_overflow_ancilla(ctrl, a, b, overflow, ancilla):
    n = len(a)
    
    if n == 1:
        #CCX | (a[0],b[0],overflow)
        CCX | (a[0],b[0],ancilla)
        CCX | (ancilla,ctrl,overflow)
        CCX | (a[0],b[0],ancilla)
        CX  | (a[0],b[0])
        return
    
    # step 1
    for i in range(n - 1):
        CX | (a[i], b[i])
    # step 2
    CCX | (ctrl, a[0], overflow)
    for i in range(n - 2):
        CX | (a[i + 1], a[i])
    # step 3
    for i in range(n - 1):
        CCX | (a[n - 1 - i], b[n - 1 - i], a[n - 2 - i])
    # step 4
    CCX | (a[0], b[0], ancilla)
    CCX | (ctrl, ancilla, overflow)
    CCX | (a[0], b[0], ancilla)
    CCX | (ctrl, a[0], b[0])
    # step 5
    for i in range(n - 1):
        CCX | (a[i + 1], b[i + 1], a[i])
        CCX | (ctrl, a[i + 1], b[i + 1])
    # step 6
    for i in range(n - 2):
        CX | (a[n - 2 - i], a[n - 3 - i])
    # step 7
    for i in range(n - 1):
        CX | (a[i], b[i])


def ctrl_add(ctrl, a, b):
    """
    (ctrl,a,b) -> (ctrl,a,b+a)
    """
    n = len(a)
    # step 1
    for i in range(n - 1):
        CX | (a[i], b[i])
    # step 2
    # CCX | (ctrl,a[0],ancilla)
    for i in range(n - 2):
        CX | (a[i + 1], a[i])
    # step 3
    for i in range(n - 1):
        CCX | (a[n - 1 - i], b[n - 1 - i], a[n - 2 - i])
    # step 4
    # CCX | (a[0],b[0],ancilla)
    # CCX | (ctrl,ancilla,overflow)
    # CCX | (a[0],b[0],ancilla)
    CCX | (ctrl, a[0], b[0])
    # step 5
    for i in range(n - 1):
        CCX | (a[i + 1], b[i + 1], a[i])
        CCX | (ctrl, a[i + 1], b[i + 1])
    # step 6
    for i in range(n - 2):
        CX | (a[n - 2 - i], a[n - 3 - i])
    # step 7
    for i in range(n - 1):
        CX | (a[i], b[i])


def mult(a, b, p, ancilla):
    """
    Multiplicant: a, b
    Product: p
    (a, b, p=0, ancilla=0) -> (a, b, a*b, ancilla=0)
    """
    n = len(a)

    if n == 1:
        CCX | (a[0],b[0],p[1])
        return
    
    #step 1
    for i in range(n):
        CCX | (b[n-1], a[n-1-i], p[2*n-1-i])
    #step 2
    for i in range(n-2):
        ctrl_add_overflow_ancilla(b[n-2-i],a,p[n-1-i:2*n-1-i],p[n-2-i],p[n-3-i])
    ctrl_add_overflow_ancilla(b[0],a,p[1:n+1],p[0],ancilla)

def division(a, b, r, ancilla):
    """
    Divided: a
    Divisor: b
    (a,b,r=0,ancilla=0) -> (a%b,b,a//b,ancilla)
    """
    n = len(a)

    for i in range(n - 1):
        # Iteration(y,b,r[i])
        y = r[i + 1:n] + a[0:i + 1]
        subtraction_overflow(b, y, ancilla)
        CX | (ancilla, r[i])
        ctrl_add(ancilla, b, y)
        CX | (r[i], ancilla)
        X | r[i]
    # Iteration(a,b,r[n-1])
    subtraction_overflow(b, a, ancilla)
    CX | (ancilla, r[n - 1])
    ctrl_add(ancilla, b, a)
    CX | (r[n - 1], ancilla)
    X | r[n - 1]


class RippleCarryAdder(Synthesis):
    @staticmethod
    def execute(n):
        """
        (a,b) -> (a,b'=a+b)

        Args:
            n(int): the bit number of a and b
        
        Quregs:
            a_q(Qureg): the qureg stores a, n qubits.
            b_q(Qureg): the qureg stores b, and stores the sum 
                        after computation, n qubits.

        reference: HIMANSHU THAPLIYAL and NAGARAJAN RANGANATHAN -
        Design of Efficient Reversible Logic Based Binary and BCD adder Circuits
        https://arxiv.org/abs/1712.02630v1
        """

        circuit = Circuit(2 * n)
        a_q = circuit([i for i in range(n)])
        b_q = circuit([i for i in range(n, 2 * n)])

        adder(a_q, b_q)
        return CompositeGate(circuit.gates)


class Multiplication(Synthesis):
    @staticmethod
    def execute(n):
        """
        (a,b,p=0,ancilla=0) -> (a,b,p=a*b,ancilla=0)

        Args:
            n(int): the bit number of a and b

        Quregs:
            a_q(Qureg): the qureg stores a, n qubits.
            b_q(Qureg): the qureg stores b, n qubits.
            p_q(Qureg): the qureg stores the product, 2n qubits.
            ancilla(Qubit): the clean ancilla qubit, 1 qubit.
        
        reference: Edgard Mu˜noz-Coreas, Himanshu Thapliya -
        T-count Optimized Design of Quantum Integer Multiplication
        https://arxiv.org/abs/1706.05113v1
        """

        circuit = Circuit(4*n + 1)
        a_q = circuit([i for i in range(n)])
        b_q = circuit([i for i in range(n, 2 * n)])
        p_q = circuit([i for i in range(2 * n, 4 * n)])
        ancilla = circuit(4 * n)
        
        mult(a_q,b_q,p_q,ancilla)

        return CompositeGate(circuit.gates)

class RestoringDivision(Synthesis):
    @staticmethod
    def execute(n):
        """
        (a,b,r=0,overflow=0) -> (a%b,b,a//b,0)

        Args:
            n(int): the bit number of a and b

        Quregs:
            a_q(Qureg): the qureg stores a, n qubits.
            b_q(Qureg): the qureg stores b, and stores the quotient
                        after computation, n qubits.
            r_q(Qureg): the qureg stores the remainder, n qubits.
            of_q(Qubit): the clean ancilla qubit, 1 qubit.

        reference: Himanshu Thapliyal, Edgard Munoz-Coreas, T. S. S. Varun, and Travis S. Humble -
        Quantum Circuit Designs of Integer Division Optimizing T-count and T-depth
        http://arxiv.org/abs/1809.09732v1
        """

        circuit = Circuit(3 * n + 1)
        a_q = circuit([i for i in range(n)])
        b_q = circuit([i for i in range(n, 2 * n)])
        r_q = circuit([i for i in range(2 * n, 3 * n)])
        of_q = circuit(3 * n)

        division(a_q, b_q, r_q, of_q)

        return CompositeGate(circuit.gates)