#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 10:55
# @Author  : Zhu Qinlin
# @File    : tmvh.py

from QuICT.core import Circuit, CX, CCX, CompositeGate, X
from ..._synthesis import Synthesis


def PeresGate(a, b, c):
    """
    (a, b, c) -> (a, a xor b, a.b xor c)

    Args:
        a(Qubit): qubit
        b(Qubit): qubit
        c(Qubit): qubit
    """
    CCX | (a, b, c)
    CX | (a, b)


def AdderOverflow(a, b, overflow):
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
        PeresGate(a, b, overflow)
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
    PeresGate(a[0], b[0], overflow)
    for i in range(n - 1):
        PeresGate(a[i + 1], b[i + 1], a[i])

    # step 5
    for i in range(n - 2):
        CX | (a[n - 2 - i], a[n - 3 - i])

    # step 6
    for i in range(n - 1):
        CX | (a[i], b[i])


def Adder(a, b):
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
        PeresGate(a[i + 1], b[i + 1], a[i])

    # step 5
    for i in range(n - 2):
        CX | (a[n - 2 - i], a[n - 3 - i])

    # step 6
    for i in range(n - 1):
        CX | (a[i], b[i])


def Subtraction(a, b):
    """
    (a,b) -> (a,b-a)
    """
    X | b
    Adder(a, b)
    X | b


def SubtractionOverflow(a, b, overflow):
    """
    (a,b) -> (a,b-a)
    """
    X | b
    X | overflow
    AdderOverflow(a, b, overflow)
    X | b
    X | overflow


"""
def CtrlAddOverflowAncilla(ctrl, a, b, overflow, ancilla):
    n = len(a)
    # step 1
    for i in range(n - 1):
        CX | (a[i], b[i])
    # step 2
    CCX | (ctrl, a[0], ancilla)
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
"""


def CtrlAdd(ctrl, a, b):
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


def Division(a, b, r, ancilla):
    """
    Divided: a
    Divisor: b
    (a,b,r=0,ancilla=0) -> (a%b,b,a//b,ancilla)
    """
    n = len(a)

    for i in range(n - 1):
        # Iteration(y,b,r[i])
        y = r[i + 1:n] + a[0:i + 1]
        SubtractionOverflow(b, y, ancilla)
        CX | (ancilla, r[i])
        CtrlAdd(ancilla, b, y)
        CX | (r[i], ancilla)
        X | r[i]
    # Iteration(a,b,r[n-1])
    SubtractionOverflow(b, a, ancilla)
    CX | (ancilla, r[n - 1])
    CtrlAdd(ancilla, b, a)
    CX | (r[n - 1], ancilla)
    X | r[n - 1]


class RippleCarryAdder(Synthesis):
    @staticmethod
    def execute(n):
        """
        (a,b) -> (a,b'=a+b)

        Args:
            n(int): the bit number of a and b

        reference: HIMANSHU THAPLIYAL and NAGARAJAN RANGANATHAN -
        Design of Efficient Reversible Logic Based Binary and BCD Adder Circuits
        https://arxiv.org/abs/1712.02630v1
        """

        circuit = Circuit(2 * n)
        qubit_a = circuit([i for i in range(n)])
        qubit_b = circuit([i for i in range(n, 2 * n)])

        Adder(qubit_a, qubit_b)
        return CompositeGate(circuit.gates)


class RestoringDivision(Synthesis):
    @staticmethod
    def execute(n):
        """
        (a,b,r=0) -> (a%b,b,a//b)

        Args:
            n(int): the bit number of a and b

        reference: Himanshu Thapliyal, Edgard Munoz-Coreas, T. S. S. Varun, and Travis S. Humble -
        Quantum Circuit Designs of Integer Division Optimizing T-count and T-depth
        http://arxiv.org/abs/1809.09732v1
        """

        circuit = Circuit(3 * n + 1)
        qubit_a = circuit([i for i in range(n)])
        qubit_b = circuit([i for i in range(n, 2 * n)])
        qubit_r = circuit([i for i in range(2 * n, 3 * n)])
        qubit_of = circuit(3 * n)

        Division(qubit_a, qubit_b, qubit_r, qubit_of)

        return CompositeGate(circuit.gates)
