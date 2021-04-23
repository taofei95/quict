#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51
# @Author  : Han Yu
# @File    : _unit_test.py

from numpy import log2, floor, gcd
from QuICT.core import *
from QuICT.qcda.synthesis import VBE
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

def test_1():
    for a in range(0, 5):
        for x in range(0, 5):
            for N in range(3, 5):
                if gcd(a, N) != 1:
                    continue
                a = a % N
                x = x % N
                n = int(floor(log2(N))) + 1
                m = 1 if x == 0 else int(floor(log2(x))) + 1
                circuit = Circuit(m + 5 * n + 2)
                qubit_x = circuit([i for i in range(m)])
                Set(qubit_x, x)
                VBE(m, a, N) | circuit
                Measure | circuit
                circuit.exec()
                if int(circuit([i for i in range(m, m + n)])) != pow(a, x) % N:
                    print(int(circuit([i for i in range(m, m + n)])))
                    print(pow(a, x) % N)
                    assert 0
    assert 1

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
