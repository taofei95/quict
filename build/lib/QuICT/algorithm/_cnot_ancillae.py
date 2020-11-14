#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/27 11:36 上午
# @Author  : Han Yu
# @File    : _cnot_ancillae.py

from ._circuit2circuit import circuit2circuit
from QuICT.models import *
from QuICT.exception import CircuitStructException
import numpy as np
from math import log2, ceil, floor

s : int
n : int
CNOT : list

def searchLayer():
    global CNOT
    length = len(CNOT)
    if length == 0:
        return -1
    return CNOT[length - 1][2]

def Copy(q, c, ql, cl, length, layer):
    global CNOT
    if length <= 0:
        return
    CNOT.append((q[ql], c[cl], layer))
    if length <= 1:
        return
    Copy(q, c, ql, cl + 1, floor(length / 2), layer + 1)
    Copy(c, c, cl, cl + floor(length / 2) + 1, floor(length / 2) - 1 + (length % 2), layer + 1)

def AddCNOT(qtemp, ql, ctemp, length, layer):
    if length <= 0:
        return
    if length == 1:
        #print('ss', qtemp[ql], ctemp)
        CNOT.append((qtemp[ql], ctemp, layer))
        return
    AddCNOT(qtemp, ql, qtemp[ql + length // 2 - 1], length // 2 - 1, layer - 1)
    AddCNOT(qtemp, ql + length // 2,  ctemp, length // 2 + (length % 2), layer - 1)
    #print('ss', qtemp[length // 2 - 1], ctemp)
    CNOT.append((qtemp[length // 2 - 1], ctemp, layer))

def ConstructU(q, initial_layer, ql, cu, sublen):
    copies = floor(pow(2, sublen) / 2)
    count = [0] * sublen
    for num in range(1, pow(2, sublen)):
        numtemp = num
        qtemp = []
        k = 0
        while numtemp != 0:
            if numtemp % 2 == 1:
                qtemp.append(q[ql + k * copies + count[k]])
                count[k] += 1
            numtemp //= 2
            k = k + 1
        layer = initial_layer + ceil(log2(sublen))
        print(cu, len(q))
        AddCNOT(qtemp, 0, q[cu], len(qtemp), layer)
        cu += 1

def StatisticalM(matrix, t, cl, subLen, col):
    sum = pow(2, subLen)
    for j in range(sum):
        t[cl + j] = 0
    for i in range(n):
        num = 0
        temp = 1
        for j in range(col, col + subLen):
            num += matrix[i][j] * temp
            temp <<= 1
        t[cl + num] += 1


def GenerateUwithX(q, c, ql, cl, blockSize, init_layer):
    length = blockSize
    subL = floor(log2(n) / 2)
    copies = int(pow(2, subL) / 2)
    print("?", length, subL, copies)
    for i in range(length):
        Copy(q, c, ql + i, cl, copies, init_layer)
        cl += copies
    initialU = cl
    initial_layer = searchLayer() + 1
    for j in range(ceil(length / subL)):
        if j > length / subL:
            subL = length - subL * floor(length / subL)
        print("??", ceil(length / subL), j,  ql + j * subL, cl)
        ConstructU(c, initial_layer, ql + j * subL, cl, subL)
        cl += 2 * copies
    endU = cl
    return initialU, endU

def GenerateVwithU(matrix, c, cl, cr,  t, blockSize, initial):
    global n
    subLen = log2(n) / 2
    block = ceil(blockSize / subLen)
    cur_col = 0
    layer = searchLayer() + 1
    for j in range(block):
        if j > blockSize / subLen:
            subLen = blockSize - subLen * floor(blockSize / subLen)
        Length = pow(2, subLen)

        StatisticalM(matrix, t, cl, subLen, initial + j * subLen)
        for k in range(1, Length):
            Copy(c, c, cl + k - 1 + j * Length, cr, ceil(t[cur_col + k]/block), layer)
            cr += ceil(t[cur_col + k] / block)
        cur_col += Length

def ConBipar(matrix, BG, t, length, cl, NewVexL, NewVexR):
    global n
    subLen = floor(log2(n) / 2)
    block = ceil(length / subLen)
    count = [0] * block * pow(2, subLen)
    sc = [0] * (2 * n)

    offset = pow(2, subLen)
    for k in range(block):
        if k != 0:
            sc[k * offset + 1] = sc[k * offset - 1] + t[k * offset - 1]
        sc[k * offset] = 0
        for i in range(2, offset):
            sc[k * offset + i] = sc[k * offset + i - 1] + t[k * offset + i - 1]
    for i in range(n):
        for j in range(block):
            num = 0
            temp = 1
            if j < length/subLen:
                subTemp = subLen
            else:
                subTemp = length - j * subLen
            for k in range(j * subLen, j * subLen + subTemp):
                num += matrix[i][cl + k] * temp
                temp *= 2
            if num == 0:
                continue
            script = sc[j * offset + num] + count[j * offset + num] / block
            BG[i][script] = True
            count[j * count + num] += 1

    DegreeL = []
    DegreeR = []
    for i in range(2 * n):
        DegreeL.append(0)
        for j in range(2 * n):
            DegreeL[i] += BG[i][j]
        if DegreeL[i] == 0:
            NewVexL.append(i)
    for j in range(2 * n):
        DegreeR.append(0)
        for i in range(n):
            DegreeR[j] += BG[i][j]
        if DegreeR[j] == 0:
            NewVexL.append(j)

    for i in range(n):
        Right = 0
        RightTemp = 0
        while (DegreeL[i] < block) and (DegreeL[i] > 0):
            if Right >= len(NewVexR):
                NewVexR[Right] = 2 * n + RightTemp
                RightTemp += 1
            if DegreeR[NewVexR[Right]] < block:
                BG[i][NewVexR[Right]] = 1
                DegreeL[i] += 1
                DegreeR[NewVexR[Right]] += 1
            Right += 1

    for j in range(2 * n):
        Left = 0
        LeftTemp = 0
        while (DegreeR[j] < block) and (DegreeR[j] > 0):
            if Left >= len(DegreeL):
                NewVexL[Left] = n + LeftTemp
                LeftTemp += 1
            if DegreeL[NewVexL[Left]] < block:
                BG[NewVexL[Left]][j] = 1
                DegreeL[NewVexL[Left]] += 1
                DegreeR[j] += 1
            Left += 1

def changeLayer(pointer, increase):
    global CNOT
    layer = increase - CNOT[pointer][2]
    CNOT[pointer] = (CNOT[pointer][0], CNOT[pointer][1], layer)

def bpm(bpGraph, m, u, seen, matchR):
    for v in range(m):
        if bpGraph[u][v] and not seen[v]:
            seen[v] = True
            if matchR[v] < 0 or bpm(bpGraph, m, matchR[v], seen, matchR):
                matchR[v] = u
                return True
    return False

def maxBPM(bpGraph, nn, m):
    matchR = [-1] * m
    result = 0
    for u in range(nn):
        seen = [False] * m
        if bpm(bpGraph, m, u, seen, matchR):
            result += 1
    return result

def Imaginary(a, img, _n):
    for i in range(_n):
        if a == img[i]:
            return True
    return False

def PerMatchLog(c, cl, z, BG, m, NewVexL, NewVexR):
    global n, CNOT
    matchR = [0] * (2 * n)
    for k in range(0, ceil(2 * log2(n)) + 2):
        layer = searchLayer() + 1
        result = maxBPM(BG, m, m)
        if result == 0:
            break
        for j in range(m):
            y = matchR[j]
            if y < 0:
                continue
            BG[y][j] = False
            if y < 0 or Imaginary(y, NewVexL, 2 * n) or Imaginary(j, NewVexR, 2 * n):
                continue

            CNOT.append((c[cl + j], z[y], layer))

    for j in range(n):
        if Imaginary(j, NewVexL, 2 * n):
            continue
        for k in range(2 * n):
            if BG[j][k] and not Imaginary(k, NewVexR, 2 * n):
                raise Exception("error")


def CNOTOneBlock(matrix, x, c, z, blockSize, initialCol, c_ini, layer):
    global CNOT
    BG = np.eye(2 * n, k = 0, dtype=bool)

    ini_pointer = len(CNOT)

    cl, cr = GenerateUwithX(x, c, initialCol, c_ini, blockSize, layer)
    t = []
    GenerateVwithU(matrix, c, t, cl,  cr, blockSize, initialCol)
    curr_pointer = len(CNOT) - 1
    curr_layer = searchLayer()
    NewLeft = []
    NewRight = []
    ConBipar(matrix, BG, t, blockSize, initialCol, NewLeft, NewRight)

    PerMatchLog(c, cr, z, BG, 2 * n, NewLeft, NewRight)
    increase = searchLayer() + curr_layer + 1
    for j in range(curr_pointer, ini_pointer - 1, -1):
        CNOT.append(CNOT[j])
        changeLayer(len(CNOT) - 1, increase)


def MainProcedure(M, x, c, z):
    global n
    global CNOT
    sumT = 0
    for l in range(s):
        blockSize = ceil(log2(n)) * ceil(log2(n))
        c_cl = 0
        layer = searchLayer() + 1
        i = 0
        while i < ceil(n / (s * blockSize)) and sumT < n:
            if sumT + blockSize > n:
                blockSize = n - sumT
            CNOTOneBlock(M, x, c, z, blockSize , sumT, c_cl, layer)
            layer = searchLayer() + 1
            sumT += blockSize
            i += 1

def InverseMatrixF2(a):
    global n
    b = np.zeros(2 * n * n).reshape(n, 2 * n)
    for i in range(n):
        for j in range(n):
            b[i][j] = a[i][j]
            b[i][j + n] = 0
        b[i][i + n] = 1
    for i in range(n):
        if b[i][i] == 0:
            flag = False
            for j in range(i + 1, n):
                if b[j][i] > 0:
                    flag = True
                    for t in range(2 * n):
                        b[j][t], b[i][t] = b[i][t], b[j][t]
            if not flag:
                return False
        for j in range(i + 1, n):
            if b[j][i] > 0:
                for t in range(2 * n):
                    b[j][t] = (b[j][t] + b[i][t]) % 2
    for i in range(n):
        for j in range(n):
            b[i][j] = b[i][j + n]
    return b

def solve(matrix):
    global CNOT, s
    CNOT = []
    x = [j for j in range(n)]
    z = [j + n for j in range(n)]
    c = [j + 2 * n for j in range(3 * s * n)]

    InvMatrix = InverseMatrixF2(matrix)

    MainProcedure(matrix, x, c, z)
    MainProcedure(InvMatrix, z, c, x)

def read(circuit : Circuit):
    global n
    n = circuit.circuit_length()
    matrix = np.identity(n, dtype=bool)
    for gate in circuit.gates:
        if gate.type() != GateType.CX:
            raise CircuitStructException("电路应当只包含CONT门")
        cindex = gate.cargs
        tindex = gate.targs
        matrix[tindex] = np.bitwise_xor(matrix[cindex], matrix[tindex])
    return matrix

class CNOT_ANCILLAE(circuit2circuit):
    @classmethod
    def run(cls, circuit : Circuit, size = 1):
        """
        :param circuit: 需变化电路
        :param size: s取值
        :return: inplace为真时,无返回值;为假时,返回新的电路,电路初值为0
        """
        global s
        s = size
        circuit.const_lock = True
        gates = cls.__run__(circuit)
        circuit.const_lock = False
        new_circuit = Circuit(len(circuit.qubits) * (2 + 3 * size))
        new_circuit.gates = gates
        return new_circuit

    @staticmethod
    def __run__(circuit : Circuit):
        matrix = read(circuit)
        solve(matrix)
        gates = []
        GateBuilder.setGateType(GateType.CX)
        for cnot in CNOT:
            GateBuilder.setCargs(cnot[0])
            GateBuilder.setTargs(cnot[1])
            gates.append(GateBuilder.getGate())
        return gates

