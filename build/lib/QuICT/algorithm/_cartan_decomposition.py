#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/4/13 7:01 下午
# @Author  : Han Yu
# @File    : _cartan_decomposition.py

from QuICT.algorithm import param2param, SyntheticalUnitary
from QuICT.models import *
import numpy as np
import copy

def v_position(vector):
    vec = vector.getA().tolist()
    l = len(vec)
    r_v = []
    for i in range(l // 2):
        r_v.append(vec[2 * i][0] * 2)
        r_v.append(0 + 0j)
    norm_v = np.linalg.norm(r_v, ord=2)
    if norm_v == 0:
        return r_v
    for i in range(l):
        r_v[i] /= norm_v
    return r_v

def v_negative(vector):
    vec = vector.getA().tolist()
    l = len(vec)
    r_v = []
    for i in range(l // 2):
        r_v.append(0 + 0j)
        r_v.append(vec[2 * i + 1][0] * 2)
    norm_v = np.linalg.norm(r_v, ord=2)
    if norm_v == 0:
        return r_v
    for i in range(l):
        r_v[i] /= norm_v
    return r_v

def complex_equal(a, b):
    return abs(a - b) < 1e-10

def complex_conj(a, b):
    return abs(a - b.conjugate()) < 1e-10

def independent(v, v_list):
    temp = copy.deepcopy(v)
    for vector in v_list:
        temp = temp - np.multiply(v, vector)
    return np.linalg.norm(temp, ord=1) > 1e-10

def uniformlyRz(low, high, y):
    if low + 1 == high:
        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setTargs(low)
        GateBuilder.setPargs(y[0] / 2)
        return [GateBuilder.getGate()]
    length = len(y) // 2
    GateBuilder.setGateType(GateType.CX)
    GateBuilder.setCargs(high - 1)
    GateBuilder.setTargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((y[i] + y[i + length]) / 2)
        Rxn.append((y[i] - y[i + length]) / 2)
    del y
    gates = uniformlyRz(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyRz(low + 1, high, Rxn))
    gates.append(gateB)
    return gates

def uniformlyRx(low, high, y):
    if low + 1 == high:
        GateBuilder.setGateType(GateType.Rx)
        GateBuilder.setTargs(low)
        GateBuilder.setPargs(y[0])
        return [GateBuilder.getGate()]
    length = len(y) // 2
    GateBuilder.setGateType(GateType.CX)
    GateBuilder.setCargs(high - 1)
    GateBuilder.setTargs(low)
    gateA = GateBuilder.getGate()
    gateB = GateBuilder.getGate()
    Rxp = []
    Rxn = []
    for i in range(length):
        Rxp.append((y[i] + y[i + length]) / 2)
        Rxn.append((y[i] - y[i + length]) / 2)
    del y
    gates = uniformlyRx(low + 1, high, Rxp)
    gates.append(gateA)
    gates.extend(uniformlyRx(low + 1, high, Rxn))
    gates.append(gateB)

    return gates

def Decomposition_Recursion(low, high, G : np.matrix):
    # S[U(2^{n-1}) ⊕ U(2^{n-1})] => SU(2^{n-1}) ⊕ SU(2^{n-1})
    # g0 ⊗ |0><0| + g1 ⊗ |1><1|
    # 1. compute M2 = LDL_dagger, M2 = g0g1_dagger ⊗ |0><0| + g1g0_dagger ⊗ |1><1|
    line = np.shape(G)[0]
    # g0 = np.mat(np.zeros((line // 2, line // 2)), dtype=complex)
    # g1 = np.mat(np.zeros((line // 2, line // 2)), dtype=complex)
    # for i in range(line // 2):
    #     row = i * 2
    #     for j in range(line // 2):
    #         col = j * 2
    #         g0[i, j] = G[row, col]
    #         g1[i, j] = G[row + 1, col + 1]
    g0 = G[:line // 2, :line // 2]
    g1 = G[line // 2:, line // 2:]

    g0dg1 = g0.H * g1
    value, u = np.linalg.eig(g0dg1)
    L = np.mat(u, dtype=complex)
    angl = []
    for v in value:
        angl.append(np.angle(v))
    gates = NKS(low, high - 1, g0 * L)
    gates.extend(uniformlyRz(low, high, angl))
    gates.extend(NKS(low, high - 1, g0))
    return gates

def SU2_to_U3(P, qubit):
    P_copy = P.copy()
    if abs(P_copy[0, 0].real) < 1e-6:
        if abs(P_copy[0, 0].imag) > 1e-6:
            P_copy[:] *= 1j
    else:
        alpha = -np.arctan(P_copy[0, 0].imag / P_copy[0, 0].real)
        phase = np.exp(alpha * 1j)
        P_copy[:] *= phase
    tha = np.arccos(P_copy[0, 0]) * 2
    if abs(tha) < 1e-6:
        thb = 0
        thc = np.angle(P_copy[1, 1])
    else:
        thc = np.angle(P_copy[0, 1] / -np.sin(tha / 2))
        thb = np.angle(P_copy[1, 0] / np.sin(tha / 2))
    GateBuilder.setGateType(GateType.U3)
    GateBuilder.setPargs([tha, thb, thc])
    GateBuilder.setTargs(qubit)
    gate = GateBuilder.getGate()
    return gate

def SO4_decomposition(P):
    detB1 = np.linalg.det(P[:2, :2])
    detB2 = np.linalg.det(P[2:, :2])
    if abs(detB1) > abs(detB2):
        B = P[:2, :2].copy()
        detB = detB1
    else:
        B = P[2:, :2].copy()
        detB = detB2
    B /= np.sqrt(detB)
    temp = np.kron(np.eye(2), B.H)
    temp = P.dot(temp)
    A = [
        [temp[0, 0], temp[0, 2]],
        [temp[2, 0], temp[2, 2]],
    ]
    detA = np.linalg.det(A)
    A /= np.sqrt(detA)

    return A, B

def takeFirst(elem):
    return elem[0]

def deal(angle, start, end):
    if start + 1 == end:
        return
    w = []
    for i in range(start, end):
        w.append(angle[i][1])
    for i in range(end - start):
        for j in range(i):
            aa = 0
            for k in range(4):
                aa += w[i][k, 0] * w[j][k, 0].conj()
            w[i] -= aa * w[j]
        aa = 0
        for j in range(4):
            aa += pow(abs(w[i][j, 0]), 2)
        aa = np.sqrt(aa)
        w[i] = w[i] / aa
    for i in range(start, end):
        angle[i] = (angle[i][0], w[i - start])

def NKS_2qubit(low, high, U : np.matrix):
    # TWO qubit decomposition https://arxiv.org/abs/0806.4015v1
    # 注意其定义的Rz与QuICT中的Rz差一个全局相位和一个负号,Ry也相差一个负号
    # print(low, high, U)
    sqrt2 = 1 / np.sqrt(2)
    B = np.mat(
        [
            [1 * sqrt2, 1j * sqrt2, 0, 0],
            [0, 0, 1j * sqrt2, 1 * sqrt2],
            [0, 0, 1j * sqrt2, -1 * sqrt2],
            [1 * sqrt2, -1j * sqrt2, 0, 0]
        ]
    )
    BH = B.H
    # 1 compute U′
    Up = BH * U * B
    # 2 compute M2
    M2 = Up.T * Up
    # 3 compute P D P_dagger

    # https://arxiv.org/pdf/quant-ph/0307190.pdf
    # compute thx, thy, thz
    # M2 = np.diag([1j, 1j, 1j, 1j])
    # M2 = np.round(M2, decimals=10)
    # print(np.round(M2, decimals=10))
    value, u = np.linalg.eig(M2)
    # print(np.round(u, decimals=2))
    u = np.mat(u)
    # print(np.round(u, decimals=2))
    # print(np.round(u * u.H))
    # print(np.linalg.det(u))
    # print(np.round(u * np.diag(value) * u.H - M2, decimals=2))
    # assert np.all(abs(u * np.diag(value) * u.T - M2) < 1e-6)

    angle = []
    for i in range(len(value)):
        angle.append((np.angle(value[i]), u[:, i]))

    angle.sort(key = takeFirst)
    last = 0
    lastTh = None
    for i in range(len(angle)):
        if lastTh is None:
            lastTh = angle[i][0]
        elif abs(angle[i][0] - lastTh) > 1e-6:
            deal(angle, last, i)
            last = i
            lastTh = angle[i][0]
    deal(angle, last, len(angle))

    # 4 compute D^(1/2) Kp

    P = np.eye(4, dtype=complex)
    for i in range(4):
        P[:, i] = angle[i][1].T
    Dd_2 = []
    Dd2 = []
    for i in range(4):
        the_angle = angle[i][0] / 2
        Dd_2.append(np.exp(-the_angle * 1j))
        Dd2.append(np.exp(the_angle * 1j))
    P = np.mat(P)
    # print(np.round(P, decimals=10))
    if np.linalg.det(P).real < 0:
        P[:, 0] = -P[:, 0]
    Dd_2 = np.diag(Dd_2)
    Dd2 = np.diag(Dd2)
    if np.linalg.det(Dd2).real < 0:
        Dd2[:, 0] = -Dd2[:, 0]
    if np.linalg.det(Dd_2).real < 0:
        Dd_2[:, 0] = -Dd_2[:, 0]
    # print(np.round(P, decimals=2))
    # print(P * P.H)
    # print(np.linalg.det(Dd2))
    # print(np.linalg.det(Dd_2))
    # print(np.round((P * Dd2 * Dd2 * P.T - M2), decimals=2))

    # assert np.all(abs(P * Dd2 * Dd2 * P.T - M2) < 1e-6)

    # 判定P \in SO(4)
    # assert abs(np.linalg.det(P) - 1) < 1e-6
    # assert np.all(abs(P * P.T - np.identity(4)) < 1e-6)

    kp = Up * P * Dd_2 * P.H

    # 判定Kp \in SO(4)
    # assert abs(np.linalg.det(kp) - 1) < 1e-6
    # assert np.all(abs(kp * kp.T - np.identity(4)) < 1e-6)

    # 判定kp * P \in SO(4)
    # ast = kp * P
    # assert abs(np.linalg.det(ast) - 1) < 1e-6
    # assert np.all(abs(ast * ast.T - np.identity(4)) < 1e-6)

    # 5 compute K1 K2
    K1 = B * kp * P * BH
    K2 = B * P.H * BH
    print(np.round(K1 * B * Dd2 * B.H * K2, decimals=2))
    print(np.round(Dd2, decimals=2))
    # print(np.round(angle[0][0], decimals=2))
    # print(np.round(angle[1][0], decimals=2))
    # print(np.round(angle[2][0], decimals=2))
    # print(np.round(angle[3][0], decimals=2))

    # thx = (angle[0][0] + angle[2][0])
    # thy = (angle[1][0] + angle[2][0])
    # thz = (angle[0][0] + angle[1][0])
    # print(thx, thy, thz)
    angle = []
    for i in range(4):
        angle.append(np.angle(Dd2[i, i]))

    thx = (angle[0] + angle[2]) / 2
    thy = (angle[1] + angle[2]) / 2
    thz = (angle[0] + angle[1]) / 2

    # 6 SO4 -> SU2 ⊗ SU2
    VA, VB = SO4_decomposition(K1)

    UA, UB = SO4_decomposition(K2)

    gates = [SU2_to_U3(UA, low), SU2_to_U3(UB, high - 1)]

    inner_gates = []

    GateBuilder.setGateType(GateType.Rz)
    GateBuilder.setTargs(high - 1)
    GateBuilder.setPargs(-np.pi / 2)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.CX)
    GateBuilder.setCargs(high - 1)
    GateBuilder.setTargs(low)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.Rz)
    GateBuilder.setTargs(low)
    GateBuilder.setPargs(-2 * thz + np.pi / 2)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.Ry)
    GateBuilder.setTargs(high - 1)
    GateBuilder.setPargs(-np.pi / 2 + 2 * thx)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.CX)
    GateBuilder.setCargs(low)
    GateBuilder.setTargs(high - 1)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.Ry)
    GateBuilder.setTargs(high - 1)
    GateBuilder.setPargs(-2 * thy + np.pi / 2)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.CX)
    GateBuilder.setCargs(high - 1)
    GateBuilder.setTargs(low)
    inner_gates.append(GateBuilder.getGate())

    GateBuilder.setGateType(GateType.Rz)
    GateBuilder.setTargs(low)
    GateBuilder.setPargs(np.pi / 2)
    inner_gates.append(GateBuilder.getGate())

    gates.extend(inner_gates)

    VA_VB = [SU2_to_U3(VA, low), SU2_to_U3(VB, high - 1)]

    gates.extend(VA_VB)

    '''
    UA_UB = gates[:2]
    VA_VB = gates[-2:]
    UA_cir = Circuit(2)
    UA_cir.set_flush_gates(UA_UB)
    VA_cir = Circuit(2)
    VA_cir.set_flush_gates(VA_VB)
    print(np.round(SyntheticalUnitary.run(VA_cir) / K1, decimals=2))
    print(np.round(SyntheticalUnitary.run(UA_cir) / K2, decimals=2))
    '''
    return gates

def NKS(low, high, g : np.matrix):
    detG = np.linalg.det(g)
    g[:] /= np.power(detG, 1.0 / np.shape(g)[0])
    if high - low == 2:
        return NKS_2qubit(low, high, g)

    # NSK algorithm https://arxiv.org/pdf/quant-ph/0509196.pdf
    # 1.Compute m2 = Theta(g_dagger)g
    Theta_g_dagger = g.H
    # Theta_g_dagger = Z^{n}g_daggerZ^{n}
    line = np.shape(Theta_g_dagger)[0]
    for i in range(line // 2):
        row = i * 2
        for j in range(line // 2):
            col = j * 2
            Theta_g_dagger[row, col+1] = -Theta_g_dagger[row, col+1]
            Theta_g_dagger[row+1, col] = -Theta_g_dagger[row+1, col]

    m2 = Theta_g_dagger * g

    # 2 Decompose m2 = pbp_dagger
    value, u = np.linalg.eig(m2)
    deal_value = [1, -1]
    deal_vector = [[], []]
    for i in range(len(value)):
        ev = value[i]
        if complex_equal(ev, 1):
            d_p = v_position(u[:, i])
            if independent(d_p, deal_vector[0]):
                deal_vector[0].append(d_p)
            d_n = v_negative(u[:, i])
            if independent(d_n, deal_vector[0]):
                deal_vector[0].append(d_n)
        elif complex_equal(ev, 0):
            d_p = v_position(u[:, i])
            if independent(d_p, deal_vector[1]):
                deal_vector[1].append(d_p)
            d_n = v_negative(u[:, i])
            if independent(d_n, deal_vector[1]):
                deal_vector[1].append(d_n)
        else:
            flag = False
            for j in range(2, len(deal_value)):
                if complex_conj(deal_value[j], ev):
                    flag = True
                    break
                elif complex_equal(deal_value[j], ev):
                    flag = True
                    deal_vector[j].append(v_position(u[:, i]))
                    deal_vector[j].append(v_negative(u[:, i]))
                    break
            if flag:
                continue
            deal_value.append(ev)
            deal_vector.append([v_position(u[:, i]), v_negative(u[:, i])])
    p = []

    angles_value = []
    for i in range(len(deal_vector)):
        vector_list = deal_vector[i]
        p.extend(vector_list)
        angl = np.angle(deal_value[i])
        for j in range(len(vector_list) // 2):
            angles_value.append(angl)
    p = np.mat(p, dtype=complex).H
    b = np.mat(np.zeros((line, line)), dtype=complex)
    # 3 find y^2 = b
    y = np.mat(np.zeros((line, line)), dtype=complex)
    for i in range(line // 2):
        angl = angles_value[i]
        b[2 * i, 2 * i] = np.cos(angl)
        b[2 * i + 1, 2 * i] = np.sin(angl) * -1j
        b[2 * i, 2 * i + 1] = np.sin(angl) * -1j
        b[2 * i + 1, 2 * i + 1] = np.cos(angl)

        y[2 * i, 2 * i] = np.cos(angl / 2)
        y[2 * i + 1, 2 * i] = np.sin(angl / 2) * -1j
        y[2 * i, 2 * i + 1] = np.sin(angl / 2) * -1j
        y[2 * i + 1, 2 * i + 1] = np.cos(angl / 2)

    # 4 compute m = pyp_dagger
    m = p * y * p.H

    # 5 computer k = gm_dagger
    k = g * m.H
    k_p = k * p

    # check
    assert not (y * y - b).any()
    assert not (m * m - p * b * p.H).any()
    assert not (m - p * y * p.H).any()
    assert not (k_p * y * p.H - g).any()
    # print(k_p * y * p.H)
    # recursion
    gates = Decomposition_Recursion(low, high, k_p)
    gates.extend(uniformlyRx(low, high, angles_value))
    gates.extend(Decomposition_Recursion(low, high, p.H))
    return gates

class Cartan_decomposition(param2param):
    @staticmethod
    def __run__(U : np.matrix) -> Circuit:
        """
        Cartan分解
        :param U: 待分解酉矩阵
        :return: 返回电路门的数组
        """
        n = np.shape(U)[0]
        high = int(np.log2(n))
        if 1 << high != n:
            raise Exception("给出的不是合法的量子电路酉矩阵")
        if (U * U.H - np.identity(n)).any():
            raise Exception("给出的不是合法的量子电路酉矩阵")
        detU = np.linalg.det(U)
        if abs(abs(detU) - 1) > 1e-6:
            raise Exception("给出的不是合法的量子电路酉矩阵")
        circuit = Circuit(high)
        gates = NKS(0, high, U)
        circuit.set_flush_gates(gates)
        return circuit

if __name__ == '__main__':
    test_circuit = Circuit(3)

    unitary = np.asmatrix([0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j,
                           0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j,
                           0.5 - 0.5j, -0.5 + 0.5j, 0. + 0.j, 0. + 0.j,
                           0. + 0.j, 0. + 0.j, 0.5 - 0.5j, -0.5 + 0.5j])
    unitary = unitary.reshape(4, 4)

    pre_unitary = SyntheticalUnitary.run(test_circuit)
    test_circuit.print_infomation()
    back_circuit = Cartan_decomposition.run(unitary)
    back_circuit.print_infomation()
    new_unitary = SyntheticalUnitary.run(back_circuit)
    print(np.round(new_unitary, decimals=2))
