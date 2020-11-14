#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/23 7:18 下午
# @Author  : Han Yu
# @File    : Shor.py.py

import random
from QuICT.models import *
from math import log, ceil, gcd, pi
from fractions import Fraction
from QuICT.algorithm import Amplitude

def run_shor(N, t = None):
    if N % 2 == 0:
        return 2

    # circuit
    n = ceil(log(N, 2))
    print(n)
    if t is None:
        t = 2 * n

    x = random.randrange(2, n)

    _gcd = gcd(x, N)
    if _gcd != 1:
       return _gcd

    circuit = Circuit(n + 1)
    X | circuit(0)
    L_bit = circuit([i for i in range(n)])
    phi_bits = [0] * t
    ancilla = circuit(n)
    for k in range(t):
        # print(Amplitude.run(circuit))
        gate_pow = pow(x, 1 << (t - 1 - k), N)
        H | ancilla
        ControlPermMul(gate_pow, N) | (ancilla, L_bit)

        for i in range(k):
            if phi_bits[i]:
                Rz(-pi / (1 << (k - i))) | L_bit
        H | ancilla

        Measure | ancilla
        circuit.flush()
        phi_bits[k] = int(ancilla)
        if phi_bits[k] == 1:
            X | ancilla

    Measure | L_bit
    y = sum([(phi_bits[t - 1 - i] * 1. / (1 << (i + 1)))
             for i in range(t)])
    # print(y)
    r = Fraction(y).limit_denominator(N - 1).denominator
    # print(x, r)
    print(r, pow(x, r // 2, N))
    if r % 2 == 0 and pow(x, r // 2, N) != N - 1:
        print(r)
        x_f1 = pow(x, r // 2) - 1
        _gcd = gcd(x_f1, N)
        if _gcd != 1:
            return _gcd
        x_z1 = pow(x, r // 2) + 1
        _gcd = gcd(x_z1, N)
        if _gcd != 1:
            return _gcd
    return 0

factor = run_shor(3939)
if factor != 0:
    print(factor)
else:
    print("Shor_fail")
