#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/23 10:14 上午
# @Author  : Han Yu
# @File    : Deutsch_Jozsa.py

from QuICT.models import Circuit, H, X, Measure, PermFx

def deutsch_jozsa_main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)

def run_deutsch_jozsa(f, n, oracle):
    '''
    自定义oracle,对一个函数f使用Deutsch_Jozsa算法判定
    :param f:       待判定的函数
    :param n:       待判定的函数的输入位数
    :param oracle:  oracle函数
    '''

    # Determine number of qreg
    circuit = Circuit(n + 1)

    # start the eng and allocate qubits
    qreg = circuit([i for i in range(n)])
    ancilla = circuit(n)

    # Start with qreg in equal superposition and ancilla in |->
    H | qreg
    X | ancilla
    H | ancilla

    # Apply oracle U_f which flips the phase of every state |x> with f(x) = 1
    oracle(f, qreg, ancilla)

    # Apply H
    H | qreg
    # Measure
    Measure | qreg
    Measure | ancilla

    circuit.flush()

    y = int(qreg)

    if y == 0:
        print('Function is constant. y={}'.format(y))
    else:
        print('Function is balanced. y={}'.format(y))

if __name__ == '__main__':
    test_number = 5
    test = [0, 1] * 2 ** (test_number - 1)
    run_deutsch_jozsa(test, test_number, deutsch_jozsa_main_oracle)
