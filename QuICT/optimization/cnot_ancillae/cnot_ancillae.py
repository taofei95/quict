#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:55 上午
# @Author  : Han Yu
# @File    : cnot_ancillae.py

from .._optimization import Optimization
from QuICT.models import *
from QuICT.exception import CircuitStructException
import numpy as np
from math import log2, ceil, floor, sqrt

s = 0
n = 0
CNOT = []

def check_matrix(M, invM):
    global n, s
    matrix = np.identity((2 + 3 * s) * n, dtype=bool)
    for cnot in CNOT:
        matrix[cnot[1]] = np.bitwise_xor(matrix[cnot[1]], matrix[cnot[0]])
    matrix = matrix[:2 * n, :2 * n]
    print(np.all(matrix[n:, :n] == M))
    MM = matrix[n:, :n]
    # print("MM", MM[0][0])
    '''
    for i in range(n):
        for j in range(n):
            if j <= i:
                if not MM[i][j]:
                    print(i, j, MM[i][j])
                    # raise Exception("error{}-{}".format(i, j))
            elif MM[i][j]:
                print(i, j,  MM[i][j])
                # raise Exception("error{}-{}".format(i, j))
    '''
    print(np.all(matrix[:n, n:] == invM))

#   添加CNOT(a, b)
def addCNOT(a, b):
    global CNOT
    CNOT.append((a, b))
    # print(a, b)

#   对一段区间[start, end)进行反向操作
def Inverse(start, end):
    # if not hasattr(Inverse, 'time'):
    #    Inverse.time = 0
    # Inverse.time += 1
    # print("INVERSE", Inverse.time, start, end, end - start)
    for i in range(end - 1, start - 1, -1):
        addCNOT(CNOT[i][0], CNOT[i][1])

#   Lemma 5中的copy过程，将x copy给c[:length]
def Copy(x, copy_list):
    own = [x]
    x_l = 1
    copy_l = len(copy_list)
    run_l = 0
    while copy_l > run_l:
        for i in range(min(x_l, copy_l - run_l)):
            addCNOT(own[i], copy_list[run_l])
            own.append(copy_list[run_l])
            run_l += 1

#   Lemma 6
def GenerateYBase(M, c, y_index, index, length, z):
    global n, s, CNOT
    ancillary = c[y_index * 3 * n + n:y_index * 3 * n + 3 * n]  # 可使用的辅助位的起点，长度为2n
    # print("ancillary start", ancillary[0], c[0])
    ystart = c[y_index * 3 * n:y_index * 3 * n + n]
    p = ancillary
    subL = floor((sqrt(length)) / 2)
    _2logn = floor(sqrt(length)) * 2
    copy_ps = ancillary[n:]
    for j in range(ceil(min(length, n - (index * s * length + y_index * length)) / subL)):
        # 构造Lemma 6中的Y_j
        M_start = index * s * length + y_index * length + j * subL  # 对应Y的位置
        sub_length = min(min(length, n - (index * s * length + y_index * length)) - j * subL, subL)
        # Step 1 Construct Pj's (Lemma 5)
        pstart = p[j * round(pow(2, subL) - 1):min((j + 1) * round(pow(2, subL) - 1), len(p))]   # 构建pj的行位置
        xstart = z[y_index * length + j * subL:]   # 对应I的行位置
        init_col = len(CNOT)
        for i in range(0, sub_length):
            copy_list = []
            for p_row in range(1, round(pow(2, sub_length))):
                if p_row & (1 << i) != 0:
                    copy_list.append(pstart[p_row - 1])
            # print(copy_list)
            Copy(xstart[i], copy_list)

        #  Step 2 Copy rows in Pj's
        tstart = ancillary[(j * (2 ** subL - 1)):]
        base = []
        for i in range(2 ** sub_length - 1):
            base.append([tstart[i]])
        temp = [0] * (2 ** sub_length - 1)
        for i in range(n):
            m_list = M[i][M_start:M_start + sub_length]
            yl = 0
            for k in range(sub_length):
                if m_list[k]:
                    yl += 1 << k
            # 优化，全0行无需复制
            if yl != 0:
                temp[yl - 1] += 1

        for i in range(2 ** sub_length - 1):
            temp[i] = ceil(temp[i] / _2logn)
            if temp[i] > 1:
                # print(j, i, temp[i] - 1, len(copy_ps))
                Copy(pstart[i], [copy_ps[i] for i in range(temp[i] - 1)])
                base[i].extend([copy_ps[j] for j in range(temp[i] - 1)])
                copy_ps = copy_ps[temp[i] - 1:]
        end_cnot = len(CNOT)
        for i in range(2 ** sub_length - 1):
            base[i] = base[i] * _2logn
        #  Step 3 Construct Y
        for i in range(n):
            m_list = M[i][M_start:]
            yl = 0
            for k in range(sub_length):
                if m_list[k]:
                    yl += 1 << k
            if yl == 0:
                continue
            yl -= 1
            if len(base[yl]) == 0:
                raise Exception("错误0!")
            addCNOT(base[yl][0], ystart[i])
            base[yl] = base[yl][1:]
        #  Step4 Restore
        Inverse(init_col, end_cnot)


#   Corollary 3
def GenerateYPart(M, x, c, index, length, z):
    global s, n
    # => R_a
    init_len = len(CNOT)
    for i in range(s):
        if index * length * s + i * length < n:
            GenerateYBase(M, c, i, index, length, x)
    end_len = len(CNOT)
    # => Add
    for i in range(s):
        for j in range(n):
            addCNOT(c[i * 3 * n + j], z[j])
    # => R_a^-1
    Inverse(init_len, end_len)

#   Lemma 4
def MainProcedure(M, x, c, z):
    global n, s
    global CNOT
    t = round(floor(log2(n)) * floor(log2(n)))
    for i in range(ceil(n / (s * t))):
        GenerateYPart(M, x[i * t * s:], c, i, t, z)

def InverseMatrixF2(a):
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

# Theorem 7
def solve(matrix):
    global CNOT, s
    CNOT = []
    x = [j for j in range(n)]
    z = [j + n for j in range(n)]
    c = [j + 2 * n for j in range(3 * s * n)]
    InvMatrix = InverseMatrixF2(matrix)
    MainProcedure(matrix, x, c, z)
    MainProcedure(InvMatrix, z, c, x)
    # check_matrix(matrix, InvMatrix)

def read(circuit : Circuit):
    global n
    n = circuit.circuit_length()
    if n < pow(log2(n), 2):
        raise CircuitStructException("电路规模n应满足n大于等于(log n)^2")
    if s > n / pow(log2(n), 2):
        raise CircuitStructException("参数s应满足s小于等于n / (log n)^2")
    matrix = np.identity(n, dtype=bool)
    for gate in circuit.gates:
        if gate.type() != GateType.CX:
            raise CircuitStructException("电路应当只包含CONT门")
        cindex = gate.cargs
        tindex = gate.targs
        matrix[tindex] = np.bitwise_xor(matrix[cindex], matrix[tindex])
    return matrix

class cnot_ancillae(Optimization):
    @classmethod
    def run(cls, circuit : Circuit, size = 1, inplace = False):
        """
        :param circuit: 需变化电路
        :param size: s取值
        :param inplace: 为真时,返回一个新的电路;为假时,修改原电路的门参数
        :return: inplace为真时,无返回值;为假时,返回新的电路,电路初值为0
        """
        global s
        s = size
        circuit.const_lock = True
        gates = cls._run(circuit)
        circuit.const_lock = False
        new_circuit = Circuit(len(circuit.qubits) * (2 + 3 * size))
        new_circuit.set_flush_gates(gates)
        return new_circuit

    @staticmethod
    def _run(circuit : Circuit, *pargs):
        matrix = read(circuit)
        solve(matrix)
        gates = []
        GateBuilder.setGateType(GateType.CX)
        for cnot in CNOT:
            GateBuilder.setCargs(cnot[0])
            GateBuilder.setTargs(cnot[1])
            gates.append(GateBuilder.getGate())
        return gates
