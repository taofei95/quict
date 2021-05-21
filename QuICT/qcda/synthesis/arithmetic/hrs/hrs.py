#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 13:12
# @Author  : Li Haomin
# @File    : hrs.py

from QuICT.core import Circuit, CompositeGate, CX, CCX, Swap, CSwap, X
from ..._synthesis import Synthesis


def EX_GCD(a, b, arr):
    """ 
    Implementation of Extended Euclidean algorithm

    Args:
        a(int): the parameter a
        b(int): the parameter b
        arr(list): store the solution of ax + by = gcd(a, b) in arr, length is 2

    """

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
    """ 
    Inversion of a in (mod N)

    Args:
        a(int): the parameter a
        n(int): the parameter n

    """
    arr = [0, 1]
    EX_GCD(a, n, arr)
    return (arr[0] % n + n) % n


def int2bitwise(c, n):
    """ 
    Transform an integer c to binary n-length bitwise string.
    
    Args:
        c(int) the parameter c
        n(int) the parameter n
    """
    c_bitwise = bin(c)[2:]
    if len(c_bitwise) > n:
        c_bitwise = c_bitwise[-n:]
        # print('c exceeds the length of a, thus is truncated')
    else:
        c_bitwise = '0' * (n - len(c_bitwise)) + c_bitwise
    return c_bitwise


def Carry(a, c_bitwise, g_aug, overflow):
    """
    Compute the overflow of a(quantum)+c(classical) with borrowed qubits g_aug.

    Args:
        a(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more bits are OK).
        overflow(Qubit): 1 qubit.
        c_bitwise(int array): n bits 0-1 array, representing binary int c.
    """
    n = len(a)
    g = g_aug[0:n - 1]
    # n==1, no borrowed bits g
    if n == 1:
        if c_bitwise[0] == '1':
            CX | (a[0], overflow)
        return
    # n>=2
    CX | (g[0], overflow)

    for i in range(n - 2):
        if c_bitwise[i] == '1':
            CX | (a[i], g[i])
            X | a[i]
        CCX | (g[i + 1], a[i], g[i])
    if c_bitwise[n - 2] == '1':
        CX | (a[n - 2], g[n - 2])
        X | a[n - 2]
    if c_bitwise[n - 1] == '1':
        CCX | (a[n - 1], a[n - 2], g[n - 2])
    for i in range(n - 2):
        CCX | (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])

    CX | (g[0], overflow)

    # uncomputation
    for i in range(n - 2):
        CCX | (g[i + 1], a[i], g[i])
    if c_bitwise[n - 1] == '1':
        CCX | (a[n - 1], a[n - 2], g[n - 2])
    if c_bitwise[n - 2] == '1':
        X | a[n - 2]
        CX | (a[n - 2], g[n - 2])
    for i in range(n - 2):
        CCX | (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])
        if c_bitwise[n - 3 - i] == '1':
            X | a[n - 3 - i]
            CX | (a[n - 3 - i], g[n - 3 - i])


def CCarry(control, a, c_bitwise, g_aug, overflow):
    """
    1-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in Carry() to CCX.

    Args:
        control(Qubit): 1 qubit.
        a(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more bits are OK).
        overflow(Qubit): 1 qubit.
        c_bitwise(int array): n bits 0-1 array, representing binary int c.
    """
    n = len(a)
    g = g_aug[0:n - 1]
    # n==1, no borrowed bits g
    if n == 1:
        if c_bitwise[0] == '1':
            CCX | (control, a[0], overflow)
        return
    # n>=2
    CCX | (control, g[0], overflow)

    for i in range(n - 2):
        if c_bitwise[i] == '1':
            CX | (a[i], g[i])
            X | a[i]
        CCX | (g[i + 1], a[i], g[i])
    if c_bitwise[n - 2] == '1':
        CX | (a[n - 2], g[n - 2])
        X | a[n - 2]
    if c_bitwise[n - 1] == '1':
        CCX | (a[n - 1], a[n - 2], g[n - 2])
    for i in range(n - 2):
        CCX | (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])

    CCX | (control, g[0], overflow)

    # uncomputation
    for i in range(n - 2):
        CCX | (g[i + 1], a[i], g[i])
    if c_bitwise[n - 1] == '1':
        CCX | (a[n - 1], a[n - 2], g[n - 2])
    if c_bitwise[n - 2] == '1':
        X | a[n - 2]
        CX | (a[n - 2], g[n - 2])
    for i in range(n - 2):
        CCX | (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])
        if c_bitwise[n - 3 - i] == '1':
            X | a[n - 3 - i]
            CX | (a[n - 3 - i], g[n - 3 - i])


def CCCarry(control1, control2, a, c_bitwise, g_aug, overflow):
    """
    2-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in Carry() to CCCX.

    Args:
        control1(Qubit): 1 qubit.
        control2(Qubit): 1 qubit.
        a(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more bits are OK).
        overflow(Qubit): 1 qubit.
        c_bitwise(int array): n bits 0-1 array, representing binary int c.
    """
    n = len(a)
    # n==1, no borrowed bits g
    if n == 1:
        if c_bitwise[0] == '1':
            # CCCX | (c1,c2,a[0],overflow) with g_aug[0] as ancilla
            CCX | (a[0], g_aug[0], overflow)
            CCX | (control1, control2, g_aug[0])
            CCX | (a[0], g_aug[0], overflow)
            CCX | (control1, control2, g_aug[0])
        return
    # n>=2
    g = g_aug[0:n - 1]
    # CCCX | (c1,c2,g[0],overflow) with a[0] as ancilla
    CCX | (g[0], a[0], overflow)
    CCX | (control1, control2, a[0])
    CCX | (g[0], a[0], overflow)
    CCX | (control1, control2, a[0])

    for i in range(n - 2):
        if c_bitwise[i] == '1':
            CX | (a[i], g[i])
            X | a[i]
        CCX | (g[i + 1], a[i], g[i])
    if c_bitwise[n - 2] == '1':
        CX | (a[n - 2], g[n - 2])
        X | a[n - 2]
    if c_bitwise[n - 1] == '1':
        CCX | (a[n - 1], a[n - 2], g[n - 2])
    for i in range(n - 2):
        CCX | (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])

    # CCCX | (c1,c2,g[0],overflow) with a[0] as ancilla
    CCX | (g[0], a[0], overflow)
    CCX | (control1, control2, a[0])
    CCX | (g[0], a[0], overflow)
    CCX | (control1, control2, a[0])

    # uncomputation
    for i in range(n - 2):
        CCX | (g[i + 1], a[i], g[i])
    if c_bitwise[n - 1] == '1':
        CCX | (a[n - 1], a[n - 2], g[n - 2])
    if c_bitwise[n - 2] == '1':
        X | a[n - 2]
        CX | (a[n - 2], g[n - 2])
    for i in range(n - 2):
        CCX | (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])
        if c_bitwise[n - 3 - i] == '1':
            X | a[n - 3 - i]
            CX | (a[n - 3 - i], g[n - 3 - i])


def SubWidget(v, g):
    """
        Subwidget used in Incrementer().

        Args:
            v(Qureg): n qubits.
            g(Qureg): n qubits(more qubits are OK).
    """
    n = len(v)
    if len(g) < n:
        print('When do Sub_Widget, no edequate ancilla qubit')

    for i in range(n - 1):
        CX | (g[n - 1 - i], v[n - 1 - i])
        CX | (g[n - 2 - i], g[n - 1 - i])
        CCX | (g[n - 1 - i], v[n - 1 - i], g[n - 2 - i])
    CX | (g[0], v[0])
    for i in range(n - 1):
        CCX | (g[i + 1], v[i + 1], g[i])
        CX | (g[i], g[i + 1])
        CX | (g[i], v[i + 1])


def Incrementer(v, g):
    """
    Incremente v by 1, with borrowed qubits g.

    Args:
        v(Qureg): n qubits.
        g(Qureg): n qubits(more qubits are OK).
    """
    n = len(v)
    if len(g) < n:
        print('When do Increment, no edequate borrowed qubit')

    for i in range(n):
        CX | (g[n - 1], v[i])
    for i in range(n - 1):
        X | g[i]
    X | v[0]
    SubWidget(v, g)
    for i in range(n - 1):
        X | g[i]
    SubWidget(v, g)
    for i in range(n):
        CX | (g[n - 1], v[i])


def CIncrementer(control, v, g_aug):
    """
    1-controlled incremente v by 1, with borrowed qubits g.

    Constructed by attaching the control qubit to the little-end of v, and apply an (n+1)-bit Incrementer() to it.

    Args:
        control(Qubit): 1 qubit.
        v(Qureg): n qubits.
        g(Qureg): n + 1 qubits(more qubits are OK).
    """
    n = len(v)
    m = len(g_aug)
    if m < n + 1:
        print("no edequate ancilla bits")
    g = g_aug[0:n + 1]
    vc = v + control
    Incrementer(vc, g)
    X | vc[n]


def Adder_rec(x, c_bitwise, ancilla, ancilla_g):
    """
    The recursively applied partial-circuit in Adder().

    Args:
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit.
        ancilla_g(Qubit): 1 qubit, might be used as borrowed qubit in CIncrementer when x_H and x_L are of the same length.
        c_bitwise(int array): n bits.
    """
    n = len(x)
    if n == 1:
        return
    mid = n // 2
    x_H = x[0:mid]
    x_L = x[mid:n]
    c_H = c_bitwise[0:mid]
    c_L = c_bitwise[mid:n]
    g = x_L + ancilla_g
    CIncrementer(ancilla, x_H, g)
    for i in range(mid):
        CX | (ancilla, x_H[i])
    Carry(x_L, c_L, x_H, ancilla)
    CIncrementer(ancilla, x_H, g)
    Carry(x_L, c_L, x_H, ancilla)
    for i in range(mid):
        CX | (ancilla, x_H[i])
    Adder_rec(x_L, c_L, ancilla, ancilla_g)
    Adder_rec(x_H, c_H, ancilla, ancilla_g)


def C_Adder_rec(control, x, c_bitwise, ancilla, ancilla_g):
    """
    The recursively applied partial-circuit in CAdder().
    
    Constructed by changing the Carry() in Adder_rec() to CCarry().

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit.
        ancilla_g(Qubit): 1 qubit, might be used as borrowed qubit in CIncrementer when x_H and x_L are of the same length.
        c_bitwise(int array): n bits.
    """
    n = len(x)
    if n == 1:
        return
    mid = n // 2
    x_H = x[0:mid]
    x_L = x[mid:n]
    c_H = c_bitwise[0:mid]
    c_L = c_bitwise[mid:n]
    g = x_L + ancilla_g
    CIncrementer(ancilla, x_H, g)
    for i in range(mid):
        CX | (ancilla, x_H[i])
    CCarry(control, x_L, c_L, x_H, ancilla)
    CIncrementer(ancilla, x_H, g)
    CCarry(control, x_L, c_L, x_H, ancilla)
    for i in range(mid):
        CX | (ancilla, x_H[i])
    C_Adder_rec(control, x_L, c_L, ancilla, ancilla_g)
    C_Adder_rec(control, x_H, c_H, ancilla, ancilla_g)


def Adder(x, c, ancilla, ancilla_g):
    """
    Compute x(quantum) + c(classical) with borrowed qubits.

    Args:
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit, borrowed ancilla.
        ancilla_g(Qubit): 1 qubit, borrowed ancilla.
        c(int): integer.
    """
    n = len(x)
    c_bitwise = int2bitwise(c, n)
    Adder_rec(x, c_bitwise, ancilla, ancilla_g)
    for i in range(n):
        if c_bitwise[i] == '1':
            X | x[i]


def CAdder(control, x, c, ancilla, ancilla_g):
    """
    Compute x(quantum) + c(classical) with borrowed qubits, 1-controlled.

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit, borrowed ancilla.
        ancilla_g(Qubit): 1 qubit, borrowed ancilla.
        c(int): integer.
    """
    n = len(x)
    c_bitwise = int2bitwise(c, n)
    C_Adder_rec(control, x, c_bitwise, ancilla, ancilla_g)
    # print(Amplitude.run(circuit))
    for i in range(n):
        if c_bitwise[i] == '1':
            CX | (control, x[i])


def CSub(control, x, c, ancilla, ancilla_g):
    """
    Compute x(quantum) - c(classical) with borrowed qubits, 1-controlled.

    Constructed on the basis of CAdder() with complement technique.

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit, borrowed ancilla.
        ancilla_g(Qubit): 1 qubit, borrowed ancilla.
        c(int): integer.
    """
    n = len(x)
    c_complement = 2 ** n - c
    cc_bitwise = int2bitwise(c_complement, n)
    C_Adder_rec(control, x, cc_bitwise, ancilla, ancilla_g)
    for i in range(n):
        if cc_bitwise[i] == '1':
            CX | (control, x[i])


def Compare(b, c, g_aug, indicator):
    """
    Compare b and c with borrowed qubits g_aug. The Indicator toggles if c > b, not if c <= b.

    Constructed on the basis of Carry().

    Args:
        b(Qureg): n qubits.
        g_aug(Qubit): n-1 qubits(more qubits are OK).
        indicator(Qubit): 1 qubit.
        c(int): integer with less-than-n length.
    """
    n = len(b)
    m = len(g_aug)
    if m < n - 1:
        print('No edequate ancilla bits when compare\n')
        return
    c_bitwise = int2bitwise(c, n)
    X | b
    Carry(b, c_bitwise, g_aug, indicator)
    X | b


def CCompare(control, b, c, g_aug, indicator):
    """
    Compare b and c with borrowed qubits g_aug. The Indicator toggles if c > b, not if c <= b, 1controlled.

    Constructed on the basis of CCarry().

    Args:
        control: 1 qubit.
        b(Qureg): n qubits.
        g_aug(Qubit): n-1 qubits(more qubits are OK).
        indicator(Qubit): 1 qubit.
        c(int): integer with less-than-n length.
    """
    n = len(b)
    m = len(g_aug)
    if m < n - 1:
        print('No edequate ancilla bits when compare\n')
        return
    c_bitwise = int2bitwise(c, n)
    X | b
    CCarry(control, b, c_bitwise, g_aug, indicator)
    X | b


def CCCompare(control1, control2, b, c, g_aug, indicator):
    """
    Compare b and c with borrowed qubits g_aug. The Indicator toggles if c > b, not if c <= b, 2controlled.

    Constructed on the basis of CCCarry().

    Args:
        control1(Qubit): 1 qubit.
        control2(Qubit): 1 qubit.
        b(Qureg): n qubits.
        g_aug(Qubit): n-1 qubits(more qubits are OK).
        indicator(Qubit): 1 qubit.
        c(int): integer with less-than-n length.
    """
    n = len(b)
    m = len(g_aug)
    if m < n - 1:
        print('No edequate ancilla bits when compare\n')
        return
    c_bitwise = int2bitwise(c, n)
    X | b
    CCCarry(control1, control2, b, c_bitwise, g_aug, indicator)
    X | b


def AdderMod(b, a, N, g, indicator):
    """
    Compute b(quantum) + a(classical) mod N(classical), with borrowed qubits g and ancilla qubit indicator.

    Args：
        b(Qreg): n qubits.
        g(Qureg): n-1 borrowed qubits(more qubits are OK).
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer less than N.
        N(int): integer.
    """
    Compare(b, N - a, g, indicator)
    CAdder(indicator, b, a, g[0:1], g[1:2])
    X | indicator
    CSub(indicator, b, N - a, g[0:1], g[1:2])
    X | indicator
    Compare(b, a, g, indicator)
    X | indicator


def AdderModReverse(b, a, N, g, indicator):
    """
    The reversed circuit of CCAdder_Mod()
    """
    AdderMod(b, N - a, N, g, indicator)


def CAdderMod(control, b, a, N, g, indicator):
    """
    Compute b(quantum) + a(classical) mod N(classical), 
    with borrowed qubits g and ancilla qubit indicator, 1-controlled.

    Args：
        control(Qubit): 1 qubit.
        b(Qreg): n qubits.
        g(Qureg): n-1 borrowed qubits(more qubits are OK).
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer less than N.
        N(int): integer.
    """
    CCompare(control, b, N - a, g, indicator)
    CAdder(indicator, b, a, g[0:1], g[1:2])
    CX | (control, indicator)
    CSub(indicator, b, N - a, g[0:1], g[1:2])
    CX | (control, indicator)
    CCompare(control, b, a, g, indicator)
    CX | (control, indicator)


def CAdderModReverse(control, b, a, N, g, indicator):
    """
    The reversed circuit of CCAdder_Mod()
    """
    CAdderMod(control, b, N - a, N, g, indicator)


def CCAdderMod(control1, control2, b, a, N, g, indicator):
    """
    Compute b(quantum) + a(classical) mod N(classical),
    with borrowed qubits g and ancilla qubit indicator, 2-controlled.

    Args：
        control1(Qubit): 1 qubit.
        control2(Qubit)：1 qubit.
        b(Qreg): n qubits.
        g(Qureg): n-1 borrowed qubits(more qubits are OK).
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer less than N.
        N(int): integer.
    """
    CCCompare(control1, control2, b, N - a, g, indicator)
    CAdder(indicator, b, a, g[0:1], g[1:2])
    CCX | (control1, control2, indicator)
    CSub(indicator, b, N - a, g[0:1], g[1:2])
    CCX | (control1, control2, indicator)
    CCCompare(control1, control2, b, a, g, indicator)
    CCX | (control1, control2, indicator)


def CCAdderModReverse(control1, control2, b, a, N, g, indicator):
    """
    The reversed circuit of CCAdder_Mod()
    """
    CCAdderMod(control1, control2, b, N - a, N, g, indicator)


# x: n bits, b: n bits
def MulModRaw(x, a, b, N, indicator):
    """
    Compute b(quantum) + x(quantum) * a(classical) mod N(classical), with target qubits b and ancilla qubit indicator.

    Args:
        x(Qureg): n qubits.
        b(Qureg): n qubits, target.
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer.
        N(int): integer.
    """
    n = len(x)
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    for i in range(n):
        # borrow all the n-1 unused qubits in x
        g = x[:n - i - 1] + x[n - i:]
        CAdderMod(x[n - 1 - i], b, a_list[i], N, g, indicator)


def MulModRawReverse(x, a, b, N, indicator):
    """
    The reversed circuit of MulModRaw()
    """
    n = len(x)
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    for i in range(n):
        g = x[:i] + x[i + 1:]
        CAdderMod(x[i], b, N - a_list[n - i - 1], N, g, indicator)


# x: n bits, b: n bits
def CMulModRaw(control, x, a, b, N, indicator):
    """
    Compute b(quantum) + x(quantum) * a(classical) mod N(classical), with target qubits b and ancilla qubit indicator,
    1-controlled.

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        b(Qureg): n qubits, target.
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer.
        N(int): integer.
    """
    n = len(x)
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    for i in range(n):
        # borrow all the n-1 unused qubits in x
        g = x[:n - i - 1] + x[n - i:]
        CCAdderMod(control, x[n - 1 - i], b, a_list[i], N, g, indicator)


def CMulModRawReverse(control, x, a, b, N, indicator):
    """
    The reversed circuit of CMulModRaw()
    """
    n = len(x)
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    for i in range(n):
        g = x[:i] + x[i + 1:]
        CCAdderMod(control, x[i], b, N - a_list[n - i - 1], N, g, indicator)

"""
def CSwap(control, a, b):
    CX | (a, b)
    CCX | (control, b, a)
    CX | (a, b)
"""

# x: n bits, ancilla: n bits, indicator: 1 bit
def MulMod(x, a, ancilla, N, indicator):
    """
    Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits.

    Args:
        x(Qureg): n qubits.
        ancilla(Qureg): n qubits.
        indicator(Qubit): 1 qubit.
        a(int): integer.
        N(int): integer.
    """
    n = len(x)
    a_r = ModReverse(a, N)
    MulModRaw(x, a, ancilla, N, indicator)
    # CSwap
    for i in range(n):
        Swap(x[i], ancilla[i])
    MulModRawReverse(x, a_r, ancilla, N, indicator)


# x: n bits, ancilla: n bits, indicator: 1 bit
def CMulMod(control, x, a, ancilla, N, indicator):
    """
    Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits, 1-controlled.

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        ancilla(Qureg): n qubits.
        indicator(Qubit): 1 qubit.
        a(int): integer.
        N(int): integer.
    """
    n = len(x)
    a_r = ModReverse(a, N)
    CMulModRaw(control, x, a, ancilla, N, indicator)
    # CSwap
    for i in range(n):
        CSwap(control, x[i], ancilla[i])
    CMulModRawReverse(control, x, a_r, ancilla, N, indicator)


def HRSIncrementerDecomposition(n):
    circuit = Circuit(n * 2)
    qubit_a = circuit([i for i in range(n)])
    qubit_g = circuit([i for i in range(n, 2 * n)])
    Incrementer(qubit_a, qubit_g)
    return CompositeGate(circuit.gates)


HRSIncrementer = Synthesis(HRSIncrementerDecomposition)


def HRSAdderDecompossition(n, c):
    circuit = Circuit(n + 2)
    qubit_x = circuit([i for i in range(n)])
    ancilla = circuit(n)
    ancilla_g = circuit(n + 1)

    Adder(qubit_x, c, ancilla, ancilla_g)

    return CompositeGate(circuit.gates)


HRSAdder = Synthesis(HRSAdderDecompossition)


def HRSCSubDecomposition(n, c):
    circuit = Circuit(n + 3)
    control = circuit(0)
    qubit_x = circuit([i for i in range(1, n + 1)])
    ancilla = circuit(n + 1)
    ancilla_g = circuit(n + 2)

    CSub(control, qubit_x, c, ancilla, ancilla_g)

    return CompositeGate(circuit.gates)


HRSCSub = Synthesis(HRSCSubDecomposition)


def HRSCCCompareDecomposition(n, c):
    circuit = Circuit(2 * n + 2)
    control1 = circuit(0)
    control2 = circuit(1)
    qubit_b = circuit([i for i in range(2, n + 2)])
    g_aug = circuit([i for i in range(n + 2, 2 * n + 1)])
    indicator = circuit(2 * n + 1)

    CCCompare(control1, control2, qubit_b, c, g_aug, indicator)

    return CompositeGate(circuit.gates)


HRSCCCompare = Synthesis(HRSCCCompareDecomposition)


def HRSAdderModDecomposition(n, a, N):
    circuit = Circuit(2 * n)
    qubit_b = circuit([i for i in range(n)])
    g = circuit([i for i in range(n, 2 * n - 1)])
    indicator = circuit(2 * n - 1)

    AdderMod(qubit_b, a, N, g, indicator)

    return CompositeGate(circuit.gates)


HRSAdderMod = Synthesis(HRSAdderModDecomposition)


def HRSCMulModRawDecomposition(n, a, N):
    circuit = Circuit(2 * n + 2)
    control = circuit(0)
    qubit_x = circuit([i for i in range(1, n + 1)])
    qubit_b = circuit([i for i in range(n + 1, 2 * n + 1)])
    indicator = circuit(2 * n + 1)

    CMulModRaw(control, qubit_x, a, qubit_b, N, indicator)

    return CompositeGate(circuit.gates)


HRSCMulModRaw = Synthesis(HRSCMulModRawDecomposition)


def HRSMulModDecomposition(n, a, N):
    circuit = Circuit(2 * n + 1)
    qubit_x = circuit([i for i in range(n)])
    ancilla = circuit([i for i in range(n, 2 * n)])
    indicator = circuit(2 * n)

    MulMod(qubit_x, a, ancilla, N, indicator)

    return CompositeGate(circuit.gates)


HRSMulMod = Synthesis(HRSMulModDecomposition)
