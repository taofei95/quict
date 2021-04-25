#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51
# @Author  : Han Yu
# @File    : _unit_test.py

from numpy import log2, floor, gcd
from QuICT.core import *
from QuICT.qcda.synthesis.tmvh import *
import pytest

def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n-1-i]
        N = N//2

def test_RestoringDivision():
    for a in range(0,20):
        for b in range(1,20):
            n = max(len(bin(a))-2,len(bin(b))-2)
            circuit = Circuit(3*n)
            a_q = circuit([i for i in range(n)])
            b_q = circuit([i for i in range(n,2*n)])
            r_q = circuit([i for i in range(2*n,3*n)])
            Set(a_q,a)
            Set(b_q,b)
            RestoringDivision(n) | circuit
            Measure | circuit
            circuit.exec()
            if int(a_q) != a%b or int(r_q) != a//b:
                print("%d // %d = %d …… %d"%(a,b,int(r_q),int(a_q)))
                assert 0
    assert 1

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
