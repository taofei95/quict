#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/5/19 10:58 上午
# @Author  : Han Yu
# @File    : Shor.py

from QuICT.models import *
from .._algorithm import Algorithm
from QuICT.algorithm import Amplitude
import numpy as np
import random
import time
from fractions import Fraction

FFT_gate = 0
oracle_gate = 0
total_FFT_gate_number = 0
total_oracle_gate_number = 0
IQFT_gate = 0

mod_multy = None
out_vector = None


def unit_test(circuit, L, string = "默认"):
    print("测试开始--", string)
    circuit.flush()
    tangle = circuit(0)[0].tangle
    for k in range(1 << (2 * L + 3)):
        if abs(tangle.values[k]) > 0.000001:
            l = []
            for ll in range(2 * L + 3):
                if (1 << ll) & k:
                    l.append(2 * L + 2 - ll)
            l.reverse()
            print(k, l, np.round(tangle.values[k], decimals=4))
    print("测试结束--", string)

def run_out(now, L, goal, nowa, out, N):
    # print(now, nowa, out)
    global out_vector
    if now + 1 == L:
        if nowa == goal:
            out_vector.append(out)
        if nowa * mod_multy[now] % N == goal:
            out_vector.append(out + 1)
        return
    run_out(now + 1, L, goal, nowa, out, N)
    run_out(now + 1, L, goal, nowa * mod_multy[now], out + (1 << (L - now - 1)), N)

def EX_GCD(a, b, arr):
    if b == 0:
        arr[0] = 1
        arr[1] = 0
        return a
    g = EX_GCD(b, a % b, arr)
    t = arr[0]
    arr[0] = arr[1]
    arr[1] = t - int(a / b) * arr[1]
    return g

def ModReverse(a, n):
    arr = [0, 1]
    EX_GCD(a, n, arr)
    return (arr[0] % n + n) % n

def fast_power(a, b, N):
    x = 1
    now_a = a
    while b > 0:
        if b % 2 == 1:
            x = x * now_a % N
        now_a = now_a * now_a % N
        b >>= 1
    return x

def controlAddMod(c1, c2, a, Nth, L, circuit):
    global FFT_gate
    an = []
    for j in range(L + 1):
        an.append(a % 2)
        a >>= 1
    an.reverse()
    th = []
    for j in range(L + 1):
        tht = 0.0
        coe = 1.0
        for k in range(j, L + 1):
            if an[k] == 1:
                tht += coe
            coe /= 2
        th.append(tht)
    th.reverse()
    for j in range(L + 1):
        CCRz(np.pi * th[j]) | circuit([c1, c2, L + 1 + j])

    for j in range(L + 1):
        Rz(-np.pi * Nth[j]) | circuit(L + 1 + j)

    temp = len(circuit.gates)
    IQFT | circuit([i for i in range(2 * L + 1, L, -1)])

    temp = len(circuit.gates) - temp
    FFT_gate += temp

    CX  | circuit([2 * L + 1, 2 * L + 2])

    temp = len(circuit.gates)
    QFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    for j in range(L + 1):
        CRz_Decompose(np.pi * Nth[j]) | circuit([2 * L + 2, L + 1 + j])

    for j in range(L + 1):
        CCRz(-np.pi * th[j]) | circuit([c1, c2, L + 1 + j])

    temp = len(circuit.gates)
    IQFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    X | circuit(2 * L + 1)
    CX | circuit([2 * L + 1, 2 * L + 2])
    X | circuit(2 * L + 1)

    temp = len(circuit.gates)
    QFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    for j in range(L + 1):
        CCRz(np.pi * th[j]) | circuit([c1, c2, L + 1 + j])

def controlAddMod_reverse(c1, c2, a, Nth, L, circuit):
    global FFT_gate
    an = []
    for j in range(L + 1):
        an.append(a % 2)
        a >>= 1
    an.reverse()
    th = []
    for j in range(L + 1):
        tht = 0.0
        coe = 1.0
        for k in range(L + 1 - j):
            if an[j + k] == 1:
                tht += coe
            coe /= 2
        th.append(tht)
    th.reverse()
    for j in range(L + 1):
        CCRz(-np.pi * th[j]) | circuit([c1, c2, L + 1 + j])

    temp = len(circuit.gates)
    IQFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    X | circuit(2 * L + 1)
    CX | circuit([2 * L + 1, 2 * L + 2])
    X | circuit(2 * L + 1)

    temp = len(circuit.gates)
    QFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    for j in range(L + 1):
        CCRz(np.pi * th[j]) | circuit([c1, c2, L + 1 + j])

    for j in range(L + 1):
        CRz_Decompose(-np.pi * Nth[j]) | circuit([2 * L + 2, L + 1 + j])

    temp = len(circuit.gates)
    IQFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    CX  | circuit([2 * L + 1, 2 * L + 2])

    temp = len(circuit.gates)
    QFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    for j in range(L + 1):
        Rz(np.pi * Nth[j]) | circuit(L + 1 + j)

    for j in range(L + 1):
        CCRz(-np.pi * th[j]) | circuit([c1, c2, L + 1 + j])

def cmult(a, N, Nth, L, circuit):
    global FFT_gate
    temp = len(circuit.gates)
    QFT  | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

    aa = a
    for i in range(L):
        controlAddMod(0, i + 1, aa, Nth, L, circuit)
        aa = aa * 2 % N

    temp = len(circuit.gates)
    IQFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp

def cmult_reverse(a, N, Nth, L, circuit):
    global FFT_gate

    temp = len(circuit.gates)
    QFT  | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp
    aa = a
    for i in range(L):
        # controlAddMod_reverse(0, i + 1, aa, Nth, L, circuit)
        controlAddMod(0, i + 1, N - aa, Nth, L, circuit)
        aa = aa * 2 % N

    temp = len(circuit.gates)
    IQFT | circuit([i for i in range(2 * L + 1, L, -1)])
    temp = len(circuit.gates) - temp
    FFT_gate += temp


def cswap(L, circuit):
    for j in range(L):
        CX              | circuit([L + 1 + j, 1 + j])
        CCX_Decompose   | circuit([0, 1 + j, L + 1 + j])
        CX | circuit([L + 1 + j, 1 + j])

def cUa(a, a_r, N,  Nth, L, circuit):
    cmult(a, N, Nth, L, circuit)
    cswap(L, circuit)
    cmult_reverse(a_r, N,  Nth, L, circuit)

def classical_cUa(a, N, L, circuit):
    # unit_test(circuit, L)
    plist = [0]
    for i in range(1, L + 1):
        plist.append(i)
    ControlPermMulDetail(a, N) | circuit(plist)
    # unit_test(circuit, L)

def Shor(N, fidelity = None):
    global FFT_gate, oracle_gate, total_FFT_gate_number, total_oracle_gate_number, mod_multy, out_vector, IQFT_gate
    """
    :param fidelity: 保真度
    :param N: 待分解的大数
    :return:
        factor: 一个因子，返回0表示分解失败
        gate_number: 单次运算门个数，为0表示没有用到量子电路
        run_time: 单次量子电路运行时间，为0表示没有用到量子电路
        total_gate_number: 运算门 个数，为0表示没有用到量子电路
        total_run_time: 量子算法总运行时间，为0表示没有用到量子电路
    """
    # 1. If N is even, return the factor 2
    if N % 2 == 0:
        return 2, 0, 0.0, 0, 0.0

    # 2. Classically determine if N = p^q
    y = np.log2(N)
    L = int(np.ceil(np.log2(N)))
    for b in range(3, L):
        x = y / b
        squeeze = np.power(2, x)
        u1 = int(np.floor(squeeze))
        u2 = int(np.ceil(squeeze))
        if fast_power(u1, b, N) == 0 or fast_power(u2, b, N) == 0:
            return b, 0, 0.0, 0, 0.0
    total_gate_number = 0
    total_run_time = 0.0

    total_FFT_gate_number = 0
    total_oracle_gate_number = 0
    total_IQFT_gate = 0

    rd = 0
    while True:
        FFT_gate = 0
        oracle_gate = 0
        # 3. Choose a random number a, 1 < a <= N - 1
        a = random.randint(2, N - 1)
        gcd = np.gcd(a, N)
        if gcd > 1:
            continue
        print("round =", rd)
        rd += 1

        # 4. Use the order-finding quantum algorithm to find the order r of a modulo N
        gate_number = 0
        run_time = 0.0

        s = random.randint(0, N - 1)
        u = fast_power(a, s, N)
        circuit = Circuit(2 * L)
        ShorInitial([a, N, u]) | circuit
        if fidelity is not None:
            circuit.fidelity = fidelity

        IQFT_gate = len(circuit.gates)
        IQFT | circuit([i for i in range(2 * L - 1, -1, -1)])
        IQFT_gate = len(circuit.gates) - IQFT_gate

        total_IQFT_gate += IQFT_gate

        gate_number += len(circuit.gates)
        gate_number -= 1
        time_start = time.time_ns()
        circuit.complete_flush()
        time_end = time.time_ns()
        run_time += time_end - time_start

        props = Amplitude.run(circuit)
        prop = [i for i in range(len(props))]
        for i in range(len(prop)):
            pos = 0
            for j in range(2 * L):
                if i & (1 << j) != 0:
                    pos += (1 << (2 * L - 1 - j))
            prop[pos] = abs(props[i])

        for i in range(0, 2 * L):
            Measure | circuit(i)

        M = 0

        gate_number += len(circuit.gates)
        time_start = time.time_ns()
        circuit.complete_flush()
        time_end = time.time_ns()
        run_time += time_end - time_start

        for i in range(0, 2 * L):
            measure = int(circuit(i))
            if measure == 1:
                M += 1.0 / (1 << (2 * L - i))
            print(i, measure, circuit(i)[0].prop)

        total_run_time += run_time
        r = Fraction(M).limit_denominator(N - 1).denominator
        total_gate_number += gate_number
        total_oracle_gate_number += oracle_gate
        total_FFT_gate_number += FFT_gate

        # 5. cal
        print(a, r, M, fast_power(a, r, N), fast_power(a, r // 2, N))
        if fast_power(a, r, N) != 1 or r % 2 == 1 or fast_power(a, r // 2, N) == N - 1:
            continue
        b = np.gcd(fast_power(a, r // 2, N) - 1, N)
        if N % b == 0 and b != 1 and N != b:
            return b, a, r, total_gate_number / rd, total_run_time / rd, total_gate_number, total_run_time, total_IQFT_gate / rd, total_IQFT_gate, total_oracle_gate_number / rd, total_oracle_gate_number, rd, prop
        c = np.gcd(fast_power(a, r // 2, N) + 1, N)
        if N % c == 0 and c != 1 and N != b:
            return c, a, r, total_gate_number / rd, total_run_time / rd, total_gate_number, total_run_time, total_IQFT_gate / rd, total_IQFT_gate, total_oracle_gate_number / rd, total_oracle_gate_number, rd, prop

class classical_shor_factoring(Algorithm):
    @staticmethod
    def __run__(n, fidelity = None):
        return Shor(n, fidelity)

if __name__ == "__main__":
    time_start = time.time_ns()
    classical_shor_factoring.run(59 * 61, 0.99)
    time_end = time.time_ns()
    print("??", time_end - time_start)
