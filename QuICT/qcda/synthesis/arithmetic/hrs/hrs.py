#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/3 13:12
# @Author  : Li Haomin
# @File    : hrs.py

from QuICT.core import CompositeGate, CX, CCX, Swap, CSwap, X
from QuICT.algorithm.quantum_algorithm.shor.utility import int2bitwise, mod_reverse
from ..._synthesis import Synthesis


def carry(gate_set, a, c_bitwise, g_aug, overflow):
    """
    Compute the overflow of a(quantum)+c(classical) with borrowed qubits g_aug.

    Args:
        a(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more bits are OK).
        overflow(Qubit): 1 qubit.
        c_bitwise(char array): n bits '0'-'1' array, representing binary int c.
    """
    n = len(a)
    g = g_aug[0:n - 1]
    # n==1, no borrowed bits g
    with gate_set:
        if n == 1:
            if c_bitwise[0] == '1':
                CX & (a[0], overflow)
            return
        # n>=2
        CX & (g[0], overflow)

        for i in range(n - 2):
            if c_bitwise[i] == '1':
                CX & (a[i], g[i])
                X & a[i]
            CCX & (g[i + 1], a[i], g[i])
        if c_bitwise[n - 2] == '1':
            CX & (a[n - 2], g[n - 2])
            X & a[n - 2]
        if c_bitwise[n - 1] == '1':
            CCX & (a[n - 1], a[n - 2], g[n - 2])
        for i in range(n - 2):
            CCX & (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])

        CX & (g[0], overflow)

        # uncomputation
        for i in range(n - 2):
            CCX & (g[i + 1], a[i], g[i])
        if c_bitwise[n - 1] == '1':
            CCX & (a[n - 1], a[n - 2], g[n - 2])
        if c_bitwise[n - 2] == '1':
            X & a[n - 2]
            CX & (a[n - 2], g[n - 2])
        for i in range(n - 2):
            CCX & (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])
            if c_bitwise[n - 3 - i] == '1':
                X & a[n - 3 - i]
                CX & (a[n - 3 - i], g[n - 3 - i])


def c_carry(gate_set, control, a, c_bitwise, g_aug, overflow):
    """
    1-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in carry() to CCX.

    Args:
        control(Qubit): 1 qubit.
        a(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more bits are OK).
        overflow(Qubit): 1 qubit.
        c_bitwise(char array): n bits '0'-'1' array, representing binary int c.
    """
    n = len(a)
    g = g_aug[0:n - 1]
    # n==1, no borrowed bits g
    with gate_set:
        if n == 1:
            if c_bitwise[0] == '1':
                CCX & (control, a[0], overflow)
            return
        # n>=2
        CCX & (control, g[0], overflow)

        for i in range(n - 2):
            if c_bitwise[i] == '1':
                CX & (a[i], g[i])
                X & a[i]
            CCX & (g[i + 1], a[i], g[i])
        if c_bitwise[n - 2] == '1':
            CX & (a[n - 2], g[n - 2])
            X & a[n - 2]
        if c_bitwise[n - 1] == '1':
            CCX & (a[n - 1], a[n - 2], g[n - 2])
        for i in range(n - 2):
            CCX & (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])

        CCX & (control, g[0], overflow)

        # uncomputation
        for i in range(n - 2):
            CCX & (g[i + 1], a[i], g[i])
        if c_bitwise[n - 1] == '1':
            CCX & (a[n - 1], a[n - 2], g[n - 2])
        if c_bitwise[n - 2] == '1':
            X & a[n - 2]
            CX & (a[n - 2], g[n - 2])
        for i in range(n - 2):
            CCX & (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])
            if c_bitwise[n - 3 - i] == '1':
                X & a[n - 3 - i]
                CX & (a[n - 3 - i], g[n - 3 - i])


def cc_carry(gate_set, control1, control2, a, c_bitwise, g_aug, overflow):
    """
    2-controlled computation the overflow of a(quantum)+c(classical) with borrowed qubits g.

    Constructed by changing the CX on overflow in carry() to CCCX.

    Args:
        control1(Qubit): 1 qubit.
        control2(Qubit): 1 qubit.
        a(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more bits are OK).
        overflow(Qubit): 1 qubit.
        c_bitwise(char array): n bits '0'-'1' array, representing binary int c.
    """
    n = len(a)
    # n==1, no borrowed bits g
    with gate_set:
        if n == 1:
            if c_bitwise[0] == '1':
                # CCCX | (c1,c2,a[0],overflow) with g_aug[0] as ancilla
                CCX & (a[0], g_aug[0], overflow)
                CCX & (control1, control2, g_aug[0])
                CCX & (a[0], g_aug[0], overflow)
                CCX & (control1, control2, g_aug[0])
            return
        # n>=2
        g = g_aug[0:n - 1]
        # CCCX | (c1,c2,g[0],overflow) with a[0] as ancilla
        CCX & (g[0], a[0], overflow)
        CCX & (control1, control2, a[0])
        CCX & (g[0], a[0], overflow)
        CCX & (control1, control2, a[0])

        for i in range(n - 2):
            if c_bitwise[i] == '1':
                CX & (a[i], g[i])
                X & a[i]
            CCX & (g[i + 1], a[i], g[i])
        if c_bitwise[n - 2] == '1':
            CX & (a[n - 2], g[n - 2])
            X & a[n - 2]
        if c_bitwise[n - 1] == '1':
            CCX & (a[n - 1], a[n - 2], g[n - 2])
        for i in range(n - 2):
            CCX & (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])

        # CCCX | (c1,c2,g[0],overflow) with a[0] as ancilla
        CCX & (g[0], a[0], overflow)
        CCX & (control1, control2, a[0])
        CCX & (g[0], a[0], overflow)
        CCX & (control1, control2, a[0])

        # uncomputation
        for i in range(n - 2):
            CCX & (g[i + 1], a[i], g[i])
        if c_bitwise[n - 1] == '1':
            CCX & (a[n - 1], a[n - 2], g[n - 2])
        if c_bitwise[n - 2] == '1':
            X & a[n - 2]
            CX & (a[n - 2], g[n - 2])
        for i in range(n - 2):
            CCX & (g[n - 2 - i], a[n - 3 - i], g[n - 3 - i])
            if c_bitwise[n - 3 - i] == '1':
                X & a[n - 3 - i]
                CX & (a[n - 3 - i], g[n - 3 - i])


def sub_widget(gate_set, v, g):
    """
    sub_widget used in incrementer().

    Args:
        v(Qureg): n qubits.
        g(Qureg): n qubits(more qubits are OK).
    """
    n = len(v)
    if len(g) < n:
        print('When do Sub_Widget, no edequate ancilla qubit')
    with gate_set:
        for i in range(n - 1):
            CX & (g[n - 1 - i], v[n - 1 - i])
            CX & (g[n - 2 - i], g[n - 1 - i])
            CCX & (g[n - 1 - i], v[n - 1 - i], g[n - 2 - i])
        CX & (g[0], v[0])
        for i in range(n - 1):
            CCX & (g[i + 1], v[i + 1], g[i])
            CX & (g[i], g[i + 1])
            CX & (g[i], v[i + 1])


def incrementer(gate_set, v, g):
    """
    Incremente v by 1, with borrowed qubits g.

    Args:
        v(Qureg): n qubits.
        g(Qureg): n qubits(more qubits are OK).
    """
    n = len(v)
    if len(g) < n:
        print('When do Increment, no edequate borrowed qubit')
    with gate_set:
        for i in range(n):
            CX & (g[n - 1], v[i])
        for i in range(n - 1):
            X & g[i]
        X & v[0]
        sub_widget(gate_set, v, g)
        for i in range(n - 1):
            X & g[i]
        sub_widget(gate_set, v, g)
        for i in range(n):
            CX & (g[n - 1], v[i])


def c_incrementer(gate_set, control, v, g_aug):
    """
    1-controlled incremente v by 1, with borrowed qubits g.

    Constructed by attaching the control qubit to the little-end of v, 
    and apply an (n+1)-bit incrementer() to it.

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
    vc = v + [control]
    with gate_set:
        incrementer(gate_set, vc, g)
        X & vc[n]


def adder_rec(gate_set, x, c_bitwise, ancilla, ancilla_g):
    """
    The recursively applied partial-circuit in adder().

    Args:
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit.
        ancilla_g(Qubit): 1 qubit, might be used as borrowed qubit in c_incrementer
            when x_H and x_L are of the same length.
        c_bitwise(char array): n bits '0'-'1' array, representing binary int c.
    """
    n = len(x)
    if n == 1:
        return
    mid = n // 2
    x_H = x[0:mid]
    x_L = x[mid:n]
    c_H = c_bitwise[0:mid]
    c_L = c_bitwise[mid:n]
    g = x_L + [ancilla_g]
    with gate_set:
        c_incrementer(gate_set, ancilla, x_H, g)
        for i in range(mid):
            CX & (ancilla, x_H[i])
        carry(gate_set, x_L, c_L, x_H, ancilla)
        c_incrementer(gate_set, ancilla, x_H, g)
        carry(gate_set, x_L, c_L, x_H, ancilla)
        for i in range(mid):
            CX & (ancilla, x_H[i])
        adder_rec(gate_set, x_L, c_L, ancilla, ancilla_g)
        adder_rec(gate_set, x_H, c_H, ancilla, ancilla_g)


def c_adder_rec(gate_set, control, x, c_bitwise, ancilla, ancilla_g):
    """
    The recursively applied partial-circuit in c_adder().

    Constructed by changing the carry() in adder_rec() to c_carry().

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        ancilla(Qubit): 1 qubit.
        ancilla_g(Qubit): 1 qubit, 
            might be used as borrowed qubit in c_incrementer
            when x_H and x_L are of the same length.
        c_bitwise(char array): n bits '0'-'1' array, representing binary int c.
    """
    n = len(x)
    if n == 1:
        return
    mid = n // 2
    x_H = x[0:mid]
    x_L = x[mid:n]
    c_H = c_bitwise[0:mid]
    c_L = c_bitwise[mid:n]
    g = x_L + [ancilla_g]
    with gate_set:
        c_incrementer(gate_set, ancilla, x_H, g)
        for i in range(mid):
            CX & (ancilla, x_H[i])
        c_carry(gate_set, control, x_L, c_L, x_H, ancilla)
        c_incrementer(gate_set, ancilla, x_H, g)
        c_carry(gate_set, control, x_L, c_L, x_H, ancilla)
        for i in range(mid):
            CX & (ancilla, x_H[i])
        c_adder_rec(gate_set, control, x_L, c_L, ancilla, ancilla_g)
        c_adder_rec(gate_set, control, x_H, c_H, ancilla, ancilla_g)


def adder(gate_set, x, c, ancilla, ancilla_g):
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
    with gate_set:
        adder_rec(gate_set, x, c_bitwise, ancilla, ancilla_g)
        for i in range(n):
            if c_bitwise[i] == '1':
                X & x[i]


def c_adder(gate_set, control, x, c, ancilla, ancilla_g):
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
    with gate_set:
        c_adder_rec(gate_set, control, x, c_bitwise, ancilla, ancilla_g)
        for i in range(n):
            if c_bitwise[i] == '1':
                CX & (control, x[i])


def c_sub(gate_set, control, x, c, ancilla, ancilla_g):
    """
    Compute x(quantum) - c(classical) with borrowed qubits, 1-controlled.

    Constructed on the basis of c_adder() with complement technique.

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
    with gate_set:
        c_adder_rec(gate_set, control, x, cc_bitwise, ancilla, ancilla_g)
        for i in range(n):
            if cc_bitwise[i] == '1':
                CX & (control, x[i])


def compare(gate_set, b, c, g_aug, indicator):
    """
    compare b and c with borrowed qubits g_aug. The Indicator toggles if c > b, not if c <= b.

    Constructed on the basis of carry().

    Args:
        b(Qureg): n qubits.
        g_aug(Qureg): n-1 qubits(more qubits are OK).
        indicator(Qubit): 1 qubit.
        c(int): integer with less-than-n length.
    """
    n = len(b)
    m = len(g_aug)
    if m < n - 1:
        print('No edequate ancilla bits when compare\n')
        return
    c_bitwise = int2bitwise(c, n)
    with gate_set:
        for i in range(n):
            X & b[i]
        carry(gate_set, b, c_bitwise, g_aug, indicator)
        for i in range(n):
            X & b[i]


def c_compare(gate_set, control, b, c, g_aug, indicator):
    """
    compare b and c with borrowed qubits g_aug.
    The Indicator toggles if c > b, not if c <= b, 1controlled.

    Constructed on the basis of c_carry().

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
    with gate_set:
        for i in range(n):
            X & b[i]
        c_carry(gate_set, control, b, c_bitwise, g_aug, indicator)
        for i in range(n):
            X & b[i]


def cc_compare(gate_set, control1, control2, b, c, g_aug, indicator):
    """
    compare b and c with borrowed qubits g_aug.
    The Indicator toggles if c > b, not if c <= b, 2controlled.

    Constructed on the basis of cc_carry().

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
    with gate_set:
        for i in range(n):
            X & b[i]
        cc_carry(gate_set, control1, control2, b, c_bitwise, g_aug, indicator)
        for i in range(n):
            X & b


def adder_mod(gate_set, b, a, N, g, indicator):
    """
    Compute b(quantum) + a(classical) mod N(classical),
    with borrowed qubits g and ancilla qubit indicator.

    Args：
        b(Qreg): n qubits.
        g(Qureg): n-1 borrowed qubits(more qubits are OK).
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer less than N.
        N(int): integer.

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """
    if len(b) <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    with gate_set:
        compare(gate_set, b, N - a, g, indicator)
        c_adder(gate_set, indicator, b, a, g[0], g[1])
        X & indicator
        c_sub(gate_set, indicator, b, N - a, g[0], g[1])
        X & indicator
        compare(gate_set, b, a, g, indicator)
        X & indicator


def adder_mod_reversed(b, a, N, g, indicator):
    """
    The reversed circuit of adder_mod()
    """
    adder_mod(b, N - a, N, g, indicator)


def c_adder_mod(gate_set, control, b, a, N, g, indicator):
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

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """
    if len(b) <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    with gate_set:
        c_compare(gate_set, control, b, N - a, g, indicator)
        c_adder(gate_set, indicator, b, a, g[0], g[1])
        CX & (control, indicator)
        c_sub(gate_set, indicator, b, N - a, g[0], g[1])
        CX & (control, indicator)
        c_compare(gate_set, control, b, a, g, indicator)
        CX & (control, indicator)


def c_adder_mod_reversed(control, b, a, N, g, indicator):
    """
    The reversed circuit of Cc_adder_Mod()
    """
    c_adder_mod(control, b, N - a, N, g, indicator)


def cc_adder_mod(gate_set, control1, control2, b, a, N, g, indicator):
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

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """
    if len(b) <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    with gate_set:
        cc_compare(gate_set, control1, control2, b, N - a, g, indicator)
        c_adder(gate_set, indicator, b, a, g[0], g[1])
        CCX | (control1, control2, indicator)
        c_sub(gate_set, indicator, b, N - a, g[0], g[1])
        CCX | (control1, control2, indicator)
        cc_compare(gate_set, control1, control2, b, a, g, indicator)
        CCX | (control1, control2, indicator)


def mul_mod_raw(gate_set, x, a, b, N, indicator):
    """
    Compute b(quantum) + x(quantum) * a(classical) mod N(classical),
    with target qubits b and ancilla qubit indicator.

    Args:
        x(Qureg): n qubits.
        b(Qureg): n qubits, target.
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer.
        N(int): integer.

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """

    n = len(x)
    if n <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = (a * 2) % N
    with gate_set:
        for i in range(n):
            # borrow all the n-1 unused qubits in x
            g = x[:n - i - 1] + x[n - i:]
            c_adder_mod(gate_set, x[n - 1 - i], b, a_list[i], N, g, indicator)


def mul_mod_raw_reversed(gate_set, x, a, b, N, indicator):
    """
    The reversed circuit of mul_mod_raw()
    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """

    n = len(x)
    if n <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    with gate_set:
        for i in range(n):
            g = x[:i] + x[i + 1:]
            c_adder_mod(gate_set, x[i], b, N - a_list[n - i - 1], N, g, indicator)


def c_mul_mod_raw(gate_set, control, x, a, b, N, indicator):
    """
    Compute b(quantum) + x(quantum) * a(classical) mod N(classical),
    with target qubits b and ancilla qubit indicator, 1-controlled.

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        b(Qureg): n qubits, target.
        indicator(Qubit): 1 ancilla qubit.
        a(int): integer.
        N(int): integer.
    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """

    n = len(x)
    if n <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    with gate_set:
        for i in range(n):
            # borrow all the n-1 unused qubits in x
            g = x[:n - i - 1] + x[n - i:]
            cc_adder_mod(gate_set, control, x[n - 1 - i], b, a_list[i], N, g, indicator)


def c_mul_mod_raw_reversed(gate_set, control, x, a, b, N, indicator):
    """
    The reversed circuit of c_mul_mod_raw()

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """

    n = len(x)
    if n <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    a_list = []
    for i in range(n):
        a_list.append(a)
        a = a * 2 % N
    with gate_set:
        for i in range(n):
            g = x[:i] + x[i + 1:]
            cc_adder_mod(gate_set, control, x[i], b, N - a_list[n - i - 1], N, g, indicator)


def mul_mod(gate_set, x, a, ancilla, N, indicator):
    """
    Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits.

    Args:
        x(Qureg): n qubits.
        ancilla(Qureg): n qubits.
        indicator(Qubit): 1 qubit.
        a(int): integer.
        N(int): integer.

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """

    n = len(x)
    if n <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    a_r = mod_reverse(a, N)
    with gate_set:
        mul_mod_raw(gate_set, x, a, ancilla, N, indicator)
        # Swap
        for i in range(n):
            Swap & (x[i], ancilla[i])
        mul_mod_raw_reversed(gate_set, x, a_r, ancilla, N, indicator)


def c_mul_mod(gate_set, control, x, a, ancilla, N, indicator):
    """
    Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits, 1-controlled.

    Args:
        control(Qubit): 1 qubit.
        x(Qureg): n qubits.
        ancilla(Qureg): n qubits.
        indicator(Qubit): 1 qubit.
        a(int): integer.
        N(int): integer.

    Note that this circuit works only when n > 2.
    So for smaller numbers, use another design.
    """

    n = len(x)
    if n <= 2:
        raise Exception(
            "The numbers should be more than 2-length to use HRS circuits.")
    a_r = mod_reverse(a, N)
    with gate_set:
        c_mul_mod_raw(gate_set, control, x, a, ancilla, N, indicator)
        # CSwap
        for i in range(n):
            CSwap & (control, x[i], ancilla[i])
        c_mul_mod_raw_reversed(gate_set, control, x, a_r, ancilla, N, indicator)

class HRSAdder(Synthesis):
    @staticmethod
    def execute(n, c):
        """
        Compute x(quantum) + c(classical) with borrowed qubits.

        Args:
            n(int): length of numbers
            c(int): the constant added to the quantum number

        Quregs:
            x(Qureg): n qubits.
            ancilla(Qubit): 1 qubit, borrowed ancilla.
            ancilla_g(Qubit): 1 qubit, borrowed ancilla.
        """

        gate_set = CompositeGate()
        qubit_x = list(range(n))
        ancilla = n
        ancilla_g = n + 1

        adder(gate_set, qubit_x, c, ancilla, ancilla_g)

        return gate_set


class HRSAdderMod(Synthesis):
    @staticmethod
    def execute(n, a, N):
        """
        Compute b(quantum) + a(classical) mod N(classical),
        with borrowed qubits g and ancilla qubit indicator.

        Args:
            n(int): length of numbers
            a(int): the constant added to the quantum number
            N(int): the modulus

        Quregs：
            b(Qreg): n qubits.
            g(Qureg): n-1 borrowed qubits(more qubits are OK).
            indicator(Qubit): 1 ancilla qubit.

        Note that this circuit works only when n > 2.
        So for smaller numbers we use another design.
        """

        if n <= 2:
            raise Exception(
                "The numbers should be more than 2-length to use HRS circuits.")
        
        gate_set = CompositeGate()
        qubit_b = list(range(n))
        g = list(range(n, 2 * n - 1))
        indicator = 2 * n - 1

        adder_mod(gate_set, qubit_b, a, N, g, indicator)

        return gate_set


class HRSMulMod(Synthesis):
    @staticmethod
    def execute(n, a, N):
        """
        Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits.

        Args:
            n(int): length of numbers
            a(int): the constant multiplied to the quantum number
            N(int): the modulus

        Quregs:
            x(Qureg): n qubits.
            ancilla(Qureg): n qubits.
            indicator(Qubit): 1 qubit.

        Note that this circuit works only when n > 2.
        So for smaller numbers we use another design.
        """

        if n <= 2:
            raise Exception(
                "The numbers should be more than 2-length to use HRS circuits.")
        
        gate_set = CompositeGate()
        qubit_x = list(range(n))
        ancilla = list(range(n, 2 * n))
        indicator = 2 * n

        mul_mod(gate_set, qubit_x, a, ancilla, N, indicator)

        return gate_set


class CHRSMulMod(Synthesis):
    @staticmethod
    def execute(n, a, N):
        """
        Compute x(quantum) * a(classical) mod N(classical), with ancilla qubits, 1-controlled.

        Args:
            n(int): length of numbers
            a(int): the constant multiplied to the quantum number
            N(int): the modulus

        Quregs:
            control(Qubit): 1 qubit.
            x(Qureg): n qubits.
            ancilla(Qureg): n qubits.
            indicator(Qubit): 1 qubit.

        Note that this circuit works only when n > 2.
        So for smaller numbers we use another design.
        """
        
        if n <= 2:
            raise Exception(
                "The numbers should be more than 2-length to use HRS circuits.")
        
        gate_set = CompositeGate()
        qubit_x = list(range(n))
        ancilla = list(range(n, 2 * n))
        indicator = 2 * n
        control = 2 * n + 1

        c_mul_mod(gate_set, control, qubit_x, a, ancilla, N, indicator)

        return gate_set
