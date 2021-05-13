#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/10/30 11:51
# @Author  : Han Yu
# @File    : _unit_test.py

from numpy import log2, floor, gcd
from QuICT.core import *
from QuICT.qcda.synthesis.arithmetic.vbe import *
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

def test_Adder():
    for a in range(0, 15):
        for b in range(0, 15):
            n = max(len(bin(a))-2, len(bin(b))-2)
            circuit = Circuit(3*n + 1)
            qubit_a = circuit([i for i in range(n)])
            qubit_b = circuit([i for i in range(n, 2*n)])
            qubit_c = circuit([i for i in range(2*n, 3*n)])
            qubit_overflow = circuit(3*n)
            Set(qubit_a, a)
            Set(qubit_b, b)
            VBEAdder(n) | (qubit_a,qubit_b,qubit_c,qubit_overflow)
            Measure | circuit
            circuit.exec()
            if int(qubit_b) != (a+b)%(2**n):
                print("%d + %d != %d"%(a,b,int(qubit_b)))
                assert 0
    assert 1

def test_AdderMod():
    for N in range(0, 10):
        for a in range(0, N):
            for b in range(0, N):
                n = len(bin(N))-2
                circuit = Circuit(4*n + 2)
                qubit_a = circuit([i for i in range(n)])
                qubit_b = circuit([i for i in range(n, 2*n)])
                qubit_c = circuit([i for i in range(2*n, 3*n)])
                qubit_overflow = circuit(3*n)
                qubit_N = circuit([i for i in range(3*n + 1, 4*n + 1)])
                qubit_t = circuit(4*n + 1)
                Set(qubit_a, a)
                Set(qubit_b, b)
                VBEAdderMod(N,n) | (qubit_a,qubit_b,qubit_c,qubit_overflow,qubit_N,qubit_t)
                Measure | circuit
                circuit.exec()
                if int(qubit_b) != (a+b)%N:
                    print("%d + %d mod %d != %d"%(a,b,N,int(qubit_b)))
                    assert 0
    assert 1

def test_MulAddMod():
    for N in range(0, 7):
        for a in range(0, N):
            for b in range(0, N):
                for x in range(0, N):
                    n = len(bin(N))-2
                    m = len(bin(N))-2
                    circuit = Circuit(4*n + m + 2)
                    qubit_x = circuit([i for i in range(m)])
                    qubit_a = circuit([i for i in range(m,n + m)])
                    qubit_b = circuit([i for i in range(n + m, 2*n + m)])
                    qubit_c = circuit([i for i in range(2*n + m, 3*n + m)])
                    qubit_overflow = circuit(3*n + m)
                    qubit_N = circuit([i for i in range(3*n + m + 1, 4*n + m + 1)])
                    qubit_t = circuit(4*n + m + 1)
                    Set(qubit_x, x)
                    Set(qubit_b, b)
                    VBEMulAddMod(a,N,n,m) | circuit
                    Measure | circuit
                    circuit.exec()
                    if int(qubit_b) != (b+a*x)%N:
                        print("%d + %d*%d mod %d != %d"%(b,a,x,N,int(qubit_b)))
                        assert 0
    assert 1

def test_Exp():
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
                VBEExpMod(a,N,n,m) | circuit
                Measure | circuit
                circuit.exec()
                if int(circuit([i for i in range(m, m + n)])) != pow(a, x) % N:
                    print(int(circuit([i for i in range(m, m + n)])))
                    print(pow(a, x) % N)
                    assert 0
    assert 1

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
