#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51
# @Author  : Han Yu
# @File    : _unit_test.py

import pytest

from QuICT.core import *
from QuICT.qcda.synthesis.arithmetic.tmvh import *


def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n - 1 - i]
        N = N // 2
"""
def test_RippleCarryAdder():
    for a in range(0,20):
        for b in range(0,20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(2*n)
            a_q = circuit([i for i in range(n)])
            b_q = circuit([i for i in range(n, 2 * n)])
            Set(a_q,a)
            Set(b_q,b)
            RippleCarryAdder.execute(n) | circuit
            Measure | circuit
            circuit.exec()
            if int(a_q) != a or int(b_q) != (a + b)%(2**n):
                print("%d + %d = %d" %(a,b,int(b_q)))
                assert 0
    assert 1
"""
def test_multiplication():
    for a in range(0,20):
        for b in range(0,20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(4*n+1)
            a_q = circuit([i for i in range(n)])
            b_q = circuit([i for i in range(n, 2 * n)])
            p_q = circuit([i for i in range(2*n, 4*n)])
            ancilla = circuit(4*n)
            Set(a_q,a)
            Set(b_q,b)
            Multiplication.execute(n) | circuit
            Measure | circuit
            circuit.exec()
            if int(a_q) != a or int(b_q) != b or int(p_q) != a*b or int(ancilla) != 0:
                print("%d * %d = %d, ancilla = %d" %(a,b,int(p_q),int(ancilla)))
                assert 0
    assert 1

"""
def test_RestoringDivision():
    for a in range(0, 20):
        for b in range(1, 20):
            n = max(len(bin(a)) - 2, len(bin(b)) - 2)
            circuit = Circuit(3*n + 1)
            a_q = circuit([i for i in range(n)])
            b_q = circuit([i for i in range(n, 2*n)])
            r_q = circuit([i for i in range(2*n, 3*n)])
            #of_q = circuit(3 * n)
            Set(a_q, a)
            Set(b_q, b)
            RestoringDivision.execute(n) | circuit
            Measure | circuit
            circuit.exec()
            if int(a_q) != a % b or int(r_q) != a // b or int(b_q) != b:
                print("%d // %d = %d …… %d" % (a, b, int(r_q), int(a_q)))
                assert 0
    assert 1
"""

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
