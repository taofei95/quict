#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:55 上午
# @Author  : Han Yu
# @File    : cnot_ancillae.py

from math import log2, ceil, floor, sqrt

import numpy as np

from .._optimization import Optimization
from QuICT.core.exception import CircuitStructException
from QuICT.core import *

s = 0
n = 0
CNOT = []

def add_CNOT(a, b):
    """ add a cont gate with control bit a and target bit b into list CNOT

    Args:
        a(int): control bit index
        b(int): target bit index
    """
    assert a != b
    global CNOT
    CNOT.append((a, b))

#   对一段区间[start, end)进行反向操作
def Inverse(start, end):
    """ apply the inverse of gate list in CNOT[start:end] on the circuit

    Args:
        start(int): the start index of interval
        end(int): the end index of interval
    """

    for i in range(end - 1, start - 1, -1):
        add_CNOT(CNOT[i][0], CNOT[i][1])

#   Lemma 5中的copy过程，将x copy给c[:length]
def Copy(x, copy_list):
    """ copy process in Lemma 5

    :param x:
    :param copy_list:
    :return:
    """
    own = [x]
    x_l = 1
    copy_l = len(copy_list)
    run_l = 0
    while copy_l > run_l:
        for i in range(min(x_l, copy_l - run_l)):
            add_CNOT(own[i], copy_list[run_l])
            own.append(copy_list[run_l])
            run_l += 1

def ConstructPj(c, x, z, sqrtn, d2logn):
    """ Apply Lemma 5 to make |x, z, 0> -> |x, (Pj)z, 0>

    Pj is the sqrt(n) * logn/2 matrix go through F_2^{logn / 2}

    Args:
        c(list<int>): the indexes of ancillae, length is sqrt(n) * log2n / 2
        x(list<int>): the indexes of first register, length is log2n / 2
        z(list<int>): the indexes of second register, length is log2n / 2
        sqrtn(int)  : the value of sqrt(n)
        d2logn(int) : the value of logn / 2
    """

    global CNOT

    t = []
    total = 0
    first = 0

    start = len(CNOT)
    for j in range(d2logn):
        tj = 0
        for i in range(1, sqrtn):
            if i & (1 << j):
                tj += 1
        total += tj
        t.append(total - 1)
        if tj > 0:
            tj -= 1
            add_CNOT(x[j], c[first])
            now = first
            first += 1
            number_a = 1
            while tj > 0:
                for i in range(number_a):
                    print(now + i, first, len(c))
                    add_CNOT(c[now + i], c[first])
                    first += 1
                    tj -= 1
                    if tj == 0:
                        break
                    add_CNOT(c[now + i], c[first])
                    first += 1
                    tj -= 1
                    if tj == 0:
                        break
    end = len(CNOT)
    for j in range(d2logn):
        for i in range(1, sqrtn):
            if i & (1 << j):
                add_CNOT(c[t[j]], z[i - 1])
                t[j] -= 1
    Inverse(start, end)


def GenerateYBase(Y_part, c, length, x):
    """ Apply Lemma 6 to copy part of Y_part in ancillary register

    We generator Y_part[:, :log^2n]
    to c[:n] with the help of c[n :  3 * n]
    time complex: \tilde{O}(n)
    depth : O(logn)

    Args:
        Y_part(np.matrix): Y_part
        c(list<int>): indexes of ancillary register,
        length: the value of log^2n
        x(list<int>): indexes of first register, length is log^2n

    """
    global n, s, CNOT

    # ancillary qubits in Lemma 6
    ancillary = c[n: 3 * n]

    # goal qubits in Lemma 6
    ystart = c[:n]

    # d2logn = floor(logn / 2)
    d2logn = int(floor(round(sqrt(length)) / 2))

    # sqrtn = sqrt(n)
    sqrtn = int(pow(2, d2logn))

    # real length
    length = min(length, np.shape(Y_part)[1])

    Step1_start = len(CNOT)
    for j in range(ceil(length / d2logn)):
        # Step 1 Construct Pj's (Lemma 5)
        cols = min(d2logn, length - j * d2logn)

        # assert cols == d2logn

        r_sqrtn = int(pow(2, cols))

        ConstructPj(ancillary[sqrtn + sqrtn * d2logn * j // 2:], x[j * d2logn:],
                    ancillary[j * (sqrtn - 1):], r_sqrtn, cols)
    Step1_end = len(CNOT)

    pointer = n
    for k in range(ceil(length / d2logn)):
        # do Step 2 and Step 3 one by one

        #  Step 2 Copy rows in Pj's
        Step2_start = len(CNOT)

        cols = min(d2logn, length - k * d2logn)

        # assert cols == d2logn

        r_sqrtn = int(pow(2, cols))

        sl        = [0] * (r_sqrtn - 1)
        sl_origin = [[] * 0 for i in range(r_sqrtn - 1)]
        for u in range(n):
            l = 0
            for i in range(cols):
                if Y_part[u, d2logn * k + i]:
                    l += 1 << i
            if l == 0:
                continue
            sl[l - 1] += 1
        for i in range(1, r_sqrtn):
            if sl[i - 1] > 0:
                total = ceil(sl[i - 1] / (4 * d2logn)) - 1
                number_a = 1
                now = k * (sqrtn - 1) + i - 1
                sl_origin[i - 1].append((4 * d2logn, now))
                while total > 0:
                    for _ in range(number_a):
                        add_CNOT(ancillary[now], ancillary[pointer])
                        sl_origin[i - 1].append((4 * d2logn, pointer))
                        pointer += 1
                        total -= 1
                        if total == 0:
                            break
                        add_CNOT(ancillary[now], ancillary[pointer])
                        sl_origin[i - 1].append((4 * d2logn, pointer))
                        pointer += 1
                        total -= 1
                        if total == 0:
                            break
                        if now == k * (sqrtn - 1) + i - 1:
                            now = pointer - 2
                        else:
                            now += 1

        Step2_end = len(CNOT)

        #   Step 3
        for u in range(n):
            l = 0
            for i in range(cols):
                if Y_part[u,  d2logn * k + i]:
                    l += 1 << i
            if l == 0:
                continue
            l -= 1
            # print(u, l, sl_origin[l])
            assert len(sl_origin[l]) > 0
            if sl_origin[l][0][0] == 0:
                sl_origin[l] = sl_origin[l][1:]
            assert len(sl_origin[l]) > 0
            sl_origin[l][0] = (sl_origin[l][0][0] - 1, sl_origin[l][0][1])
            add_CNOT(ancillary[sl_origin[l][0][1]], ystart[u])


        #   Step 4.1 restore of Step2
        Inverse(Step2_start, Step2_end)

    # Step 4.2 restore of Step1
    Inverse(Step1_start, Step1_end)

def GenerateYPart(Y_part, x, c, index, length, z):
    """ apply Corollary 3 to make |x, z, 0> -> |x, (T_part)z, 0>

    time complex: \tilde{O}(sn)
    depth : O(logn)

    Args:
        Y_part(np.matrix): n * n, matrix in F_2. We only use the part
                           Y_part[: , :slog^2n]
        x(list<int>): the indexes of first register, we only use x[:slog^2n]
        c(list<int>): the indexes of ancillary register, length is 3 * s * n
        index(int): the index of the Y part
        length(int): the reality value for log^2n
        z(list<int>): the indexes of second register, length is n
    """
    global s, n
    # operate R_a in Corollary 3
    init_len = len(CNOT)
    for i in range(s):
        if index * length * s + i * length < n:
            GenerateYBase(Y_part[:, i * length:], c[i * 3 * n:], length, x[i * length:])
    end_len = len(CNOT)

    # operate Add in Corollary 3
    for i in range(s):
        for j in range(n):
            add_CNOT(c[i * 3 * n + j], z[j])

    # operate R_a^{-1} in Corollary 3
    Inverse(init_len, end_len)

def MainProcedure(M, x, c, z):
    """ apply Lemma4 to make |x, z, 0> -> |x, z xor Mx, 0>

    time complex: \tilde{O}(n^2)
    depth : O(n/slogn)

    Args:
        M(np.matrix): a matrix in F_2
        x(list<int>): the indexes of first register, length is n
        c(list<int>): the indexes of ancillary register, length is n
        z(list<int>): the indexes of second register, length is 3 * s * n
    """
    global n, s
    global CNOT

    # divide the part into size slog^2n and apply Lemma 5
    # we ensure the log2n be even and 2 (sqrt(n) - 1) log2n <= n
    log2n = floor(log2(n))
    while log2n % 2 == 1 or 2 * (round(pow(2, log2n / 2)) - 1) * log2n > n:
        log2n -= 1
    t = round(log2n * log2n)
    for i in range(ceil(n / (s * t))):
        GenerateYPart(M[:, i * s * t:], x[i * t * s:], c, i, t, z)
def InverseMatrixF2(a):
    """ get the inverse matrix of a in F_2

    Args:
        a(np.matrix): the matrix in F_2

    Returns:
        np.matrix: the inverse of a in F_2

    """

    global n
    b = np.zeros(2 * n * n, dtype=bool).reshape(n, 2 * n)
    for i in range(n):
        for j in range(n):
            b[i][j] = a[i][j]
            b[i][j + n] = False
        b[i][i + n] = True
    for i in range(n):
        if not b[i][i]:
            flag = False
            for j in range(i + 1, n):
                if b[j][i]:
                    flag = True
                    for t in range(2 * n):
                        b[j][t], b[i][t] = b[i][t], b[j][t]
            if not flag:
                return False
        for j in range(i + 1, n):
            if b[j][i]:
                for t in range(2 * n):
                    b[j][t] = b[j][t] ^ b[i][t]
    for i in range(n):
        for j in range(n):
            b[i][j] = b[i][j + n]
    return b[:, :n]

def solve(matrix):
    """ apply Theorem 7 to build new circuit

    Args:
        matrix(np.array): the matrix represents the CNOT circuit
    """

    global CNOT, s
    CNOT = []

    # x, z and c mean three part in the circuit in the theorem 7
    x: list = [j for j in range(n)]
    z: list = [j + n for j in range(n)]
    c: list = [j + 2 * n for j in range(3 * s * n)]
    InvMatrix = InverseMatrixF2(matrix)

    # do C_1 operate
    MainProcedure(matrix, x, c, z)

    # do C_2 operate
    MainProcedure(InvMatrix, z, c, x)

def read(circuit : Circuit):
    """ transform the CNOT circuit into a matrix with 0 and 1

    apply xor operator in an identity matrix, the cnot list can be represented by a matrix with 0 and 1

    Args:
        circuit(Circuit): CNOT circuit

    Returns:
        np.matrix: the matrix transformed by the circuit
    """
    global n
    n = circuit.circuit_length()
    if n < 4:
        raise CircuitStructException("the qubit number of circuit n \
                should greater than or equal 4")
    matrix = np.identity(n, dtype=bool)
    for gate in circuit.gates:
        if gate.type() != GateType.CX:
            raise CircuitStructException(f"the input circuit should be a CNOT circuit, but it contains {str(gate)}")
        cindex = gate.cargs
        tindex = gate.targs
        matrix[tindex] = np.bitwise_xor(matrix[cindex], matrix[tindex])
    return matrix

class cnot_ancillae(Optimization):
    @classmethod
    def run(cls, circuit : Circuit, size = 1, inplace = False):
        """ Optimization the circuit by (3s+1)n ancillary qubits

        Optimal Space-Depth Trade-Off of CNOT Circuits in Quantum Logic Synthesis
        Theorem 7
        parallelize any n-qubit CNOT circuit into O(n/slogn) depth with (3s+1) ancillae,
        where 1 <= s <= O(n/log^2n). time complex is \tilde{O}(n^w), which is time for get inverse of a matrix.
        https://arxiv.org/pdf/1907.05087.pdf

        Args:
            circuit(Circuit): circuit to be optimize
            size(int):        the 's' in the Theorem
            inplace(bool):    change the old circuit if it is true, otherwise create a new circuit
                              Note that the old circuit should have (3s + 2)n qubits with
                              first n qubits as cnot circuit, followed (3s + 1)n unused qubits
        """

        global s
        s = size
        circuit.const_lock = True
        gates = cls._run(circuit)
        circuit.const_lock = False
        new_circuit = Circuit(circuit.circuit_length() * (2 + 3 * size))
        new_circuit.set_flush_gates(gates)
        return new_circuit

    @staticmethod
    def _run(circuit : Circuit, *pargs):
        """
        Args:
            circuit(Circuit): circuit to be optimize
            *pargs: empty
        """

        matrix = read(circuit)
        solve(matrix)
        gates = []
        GateBuilder.setGateType(GateType.CX)
        for cnot in CNOT:
            GateBuilder.setCargs(cnot[0])
            GateBuilder.setTargs(cnot[1])
            gates.append(GateBuilder.getGate())
        return gates
