#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 10:55 上午
# @Author  : Han Yu
# @File    : VBE.py

from .._synthesis import Synthesis
from numpy import log2, floor, gcd
from QuICT.models import Circuit, H, X, CX, CCX, Measure, PermFx, Swap
from QuICT.algorithm import SyntheticalUnitary, Amplitude


def Inverse(a, N):
    """
    Inversion of a in (mod N)
    """
    for i in range(N):
        if i * a % N == 1:
            return i
    return None


def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    str = bin(N)[2:]
    n = len(qreg);
    m = len(str)
    if m > n:
        print('When set qureg as N=%d, N exceeds the length of qureg n=%d, thus is truncated' % (N, n))

    for i in range(min(n, m)):
        if str[m - 1 - i] == '1':
            X | qreg[n - 1 - i]


def ControlSet(control, qreg, N):
    """
    Set the qreg as N, using CX gates on specific qubits
    """
    str = bin(N)[2:]
    n = len(qreg);
    m = len(str)
    if m > n:
        print('When cset qureg as N=%d, N exceeds the length of qureg n=%d, thus is truncated' % (N, n))

    for i in range(min(n, m)):
        if str[m - 1 - i] == '1':
            CX | (control, qreg[n - 1 - i])


def CControlSet(control1, control2, qreg, N):
    """
    Set the qreg as N, using CCX gates on specific qubits
    """
    str = bin(N)[2:]
    n = len(qreg);
    m = len(str)
    if m > n:
        print('When ccset qureg as N=%d, N exceeds the length of qureg n=%d, thus is truncated' % (N, n))

    for i in range(min(n, m)):
        if str[m - 1 - i] == '1':
            CCX | (control1, control2, qreg[n - 1 - i])


# (c_in,a,b,c_out=0) -> (c_in,a,b,c_out')
# 计算单bit加法的进位,c_out'存储进位结果
def Carry(c_in, a, b, c_out):
    """
    Carry for one bit plus
    """
    CCX | (a, b, c_out)
    CX | (a, b)
    CCX | (c_in, b, c_out)


# Carry电路的逆
def ReverseCarry(c_in, a, b, c_out):
    CCX | (c_in, b, c_out)
    CX | (a, b)
    CCX | (a, b, c_out)


# (c_in,a,b) -> (c_in,a,b'=a+b+c_in)
# 计算单bit的求和运算
def Sum(c_in, a, b):
    CX | (a, b)
    CX | (c_in, b)


# Sum电路的逆
def ReverseSum(c_in, a, b):
    CX | (c_in, b)
    CX | (a, b)


# (a,b,c=0,overflow=0) -> (a,b'=a+b,c,overflow')
# 计算a+b的电路，b存储计算结果
# c是辅助bit，占n位，用来存储进位信息
# overflow：辅助比特，占1位，用来存储溢出信息
def PlainAdder(a, b, c, overflow):
    n = len(a)

    for i in range(n - 1):
        Carry(c[n - 1 - i], a[n - 1 - i], b[n - 1 - i], c[n - 2 - i])
    Carry(c[0], a[0], b[0], overflow)

    CX | (a[0], b[0])
    Sum(c[0], a[0], b[0])

    for i in range(n - 1):
        ReverseCarry(c[1 + i], a[1 + i], b[1 + i], c[i])
        Sum(c[1 + i], a[1 + i], b[1 + i])


# PlainAdder电路的逆
def ReversePlainAdder(a, b, c, overflow):
    n = len(a)

    for i in range(n - 1):
        ReverseSum(c[n - 1 - i], a[n - 1 - i], b[n - 1 - i])
        Carry(c[n - 1 - i], a[n - 1 - i], b[n - 1 - i], c[n - 2 - i])

    ReverseSum(c[0], a[0], b[0])
    CX | (a[0], b[0])

    ReverseCarry(c[0], a[0], b[0], overflow)
    for i in range(n - 1):
        ReverseCarry(c[1 + i], a[1 + i], b[1 + i], c[i])


# (a,b,c=0,overflow=0,qubit_N=0,t=0) -> (a,b'=(a+b)mod N,c,overflow',qubit_N=0,t=0)
# 计算(a+b)mod N的电路，b存储计算结果
# c：辅助bit，占n位，继承自PlainAdder
# overflow：辅助bit，占1位，继承自PlainAdder
# qubit_N：辅助比特，占n位，在计算过程中被置为N的值，
# t：辅助比特，占1位，临时备份overflow，以作uncomputation
def AdderMod(N, a, b, c, overflow, qubit_N, t):
    n = len(qubit_N)

    Set(qubit_N, N)
    PlainAdder(a, b, c, overflow)
    for i in range(n):
        Swap | (a[i], qubit_N[i])
    ReversePlainAdder(a, b, c, overflow)
    X | overflow
    CX | (overflow, t)
    X | overflow
    ControlSet(t, a, N)
    PlainAdder(a, b, c, overflow)
    ControlSet(t, a, N)
    for i in range(n):
        Swap | (a[i], qubit_N[i])
    ReversePlainAdder(a, b, c, overflow)
    CX | (overflow, t)
    PlainAdder(a, b, c, overflow)
    Set(qubit_N, N)


def ReverseAdderMod(N, a, b, c, overflow, qubit_N, t):
    n = len(qubit_N)

    Set(qubit_N, N)
    ReversePlainAdder(a, b, c, overflow)
    CX | (overflow, t)
    PlainAdder(a, b, c, overflow)
    for i in range(n):
        Swap | (a[i], qubit_N[i])
    ControlSet(t, a, N)
    ReversePlainAdder(a, b, c, overflow)
    ControlSet(t, a, N)
    X | overflow
    CX | (overflow, t)
    X | overflow
    PlainAdder(a, b, c, overflow)
    for i in range(n):
        Swap | (a[i], qubit_N[i])
    ReversePlainAdder(a, b, c, overflow)
    Set(qubit_N, N)

# (control,x,qubit_a=0,b=0,c=0,overflow=0,qubit_N=0,t=0) -> (control,x,qubit_a=0,b'=(a**control)*x mod N,c,overflow',qubit_N,t)
# 受控求模乘法电路，计算(a**control)*x mod N，b存储计算结果
# control：控制位
# qubit_a：辅助bit，占n位，继承自AdderMod；计算时被受控地相继置为a、2a、4a、...、(2**n)a，并输入AdderMod的第一个加数位置
# b：占n位，继承自AdderMod；被输入AdderMod的第二个加数位置
# c：辅助bit，占n位，继承自AdderMod
# overflow：辅助bit，占1位，继承自AdderMod
# qubit_N：辅助比特，占n位，在计算过程中被置为N的值，继承自AdderMod
# t：辅助比特，占1位，继承自AdderMod
def ControlMulMod(a, N, control, x, qubit_a, b, c, overflow, qubit_N, t):
    n = len(qubit_N)

    for i in range(n):
        CControlSet(control, x[n - 1 - i], qubit_a, a)
        AdderMod(N, qubit_a, b, c, overflow, qubit_N, t)
        CControlSet(control, x[n - 1 - i], qubit_a, a)
        a = (a * 2) % N

    X | control
    for i in range(n):
        CCX | (control, x[i], b[i])
    X | control


def ReverseControlMulMod(a, N, control, x, qubit_a, b, c, overflow, qubit_N, t):
    n = len(qubit_N)

    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N

    X | control
    for i in range(n):
        CCX | (control, x[i], b[i])
    X | control

    for i in range(n):
        CControlSet(control, x[i], qubit_a, a_list[n - 1 - i])
        ReverseAdderMod(N, qubit_a, b, c, overflow, qubit_N, t)
        CControlSet(control, x[i], qubit_a, a_list[n - 1 - i])

# (x,result=1,qubit_a=0,b=0,c=0,overflow=0,qubit_N=0,t=0)->(x,result'=(a**x)mod N,qubit_a=0,b=0,c=0,overflow',qubit_N=0,t=0)
# 计算(a**x)mod N，result存储计算结果
# x，占m位
# result，占n位，继承自ControlMulMod
# qubit_a：辅助bit，占n位，继承自ControlMulMod
# b：辅助bit，占n位，继承自ControlMulMod
# c：辅助bit，占n位，继承自ControlMulMod
# overflow：辅助bit，占1位，继承自ControlMulMod
# qubit_N：辅助bit，占n位，继承自ControlMulMod
# t：辅助bit，占1位，继承自ControlMulMod
def ExpMod(a, N, x, result, qubit_a, b, c, overflow, qubit_N, t):
    m = len(x)
    n = len(qubit_N)
    a_inv = Inverse(a, N)
    # print("a_inv=",a_inv)

    for i in range(m):
        ControlMulMod(a, N, x[m - 1 - i], result, qubit_a, b, c, overflow, qubit_N, t)
        for j in range(n):
            Swap | (result[j], b[j])
        ReverseControlMulMod(a_inv, N, x[m - 1 - i], result, qubit_a, b, c, overflow, qubit_N, t)
        a = (a ** 2) % N
        a_inv = (a_inv ** 2) % N


class VBEModel(Synthesis):
    def __call__(self, m, a, N):
        self.pargs = [m, a, N]
        return self

    def build_gate(self):
        m = self.pargs[0]
        a = self.pargs[1]
        N = self.pargs[2]

        if N <= 2:
            raise Exception("模数N应大于2")
        if gcd(a, N) != 1:
            raise Exception("a与N应当互质")
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
        ExpMod(a, N, qubit_x, qubit_r, qubit_a, qubit_b, qubit_c, overflow, qubit_N, t)

        return circuit

VBE = VBEModel()
