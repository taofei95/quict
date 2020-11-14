#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/4/13 7:01 下午
# @Author  : Han Yu
# @File    : _cartan_decomposition.py

from QuICT.synthesis import Synthesis
from QuICT.algorithm import SyntheticalUnitary
from QuICT.models import *
import numpy as np
import copy
from scipy import linalg
from scipy.stats import ortho_group
import random
EPS = 1e-14
EPS_bit = 14

def schmidt_orthogonalize(vectors):
    for i in range(4):
        vnow = vectors[:, i].copy()
        for j in range(i):
            vnow -= (vectors[:, i].T * vectors[:, j])[0, 0] * vectors[:, j]
        vnow[:] /= np.sqrt((vnow.T * vnow)[0, 0])
        vectors[:, i] = vnow

def diagonalize(small):
    """
    :param small: \in R(n) symmetric
    :return: T \in SO(4)
             D diag_list
    """
    value, P = np.linalg.eig(small)
    sort_helper = [(value[i], P[:, i].copy()) for i in range(4)]
    sort_helper.sort(key=lambda x: x[0])
    for i in range(len(small)):
        value[i] = sort_helper[i][0]
        P[:, i] = sort_helper[i][1]
    schmidt_orthogonalize(P)
    return P, value

def simultaneous_diagonalize(A, B):
    """
    :param A /in R(4) symmetric
    :param B /in R(4) symmetric
        AB = BA,
    :return:
        T /in SO(4)
        D1 diag_list
        D2 diag_list
        (T.T)AT = D1
        (T.T)BT = D2
    """
    print(np.linalg.det(A))
    print(np.linalg.det(B))
    value, P = np.linalg.eig(A)
    print(np.linalg.det(P))
    sort_helper = [(value[i], P[:, i].copy()) for i in range(4)]
    sort_helper.sort(key=lambda x: x[0])
    for i in range(4):
        value[i] = sort_helper[i][0]
        P[:, i] = sort_helper[i][1]
    print(np.linalg.det(P))
    schmidt_orthogonalize(P)
    B1 = P.T * B * P
    last = None
    start = 0
    end = 0
    Q = np.mat(np.zeros((4, 4)), dtype=np.int)
    QD = []
    for i in range(4):
        if last is None or abs(last - value[i]) < EPS:
            end += 1
        else:
            last = value[i]
            sT, sD = diagonalize(B1[start:end, start:end])
            Q[start:end, start:end] = sT
            QD.extend(sD)
            start = end
    if start != end:
        sT, sD = diagonalize(B1[start:end, start:end])
        Q[start:end, start:end] = sT
        QD.extend(sD)
    return P * Q, value, QD


def orthogonality_diagonalize(u):
    """
    :param u: u /in SU(4)
    :return: A /in SU(4) make Au(u.H)(A.H) a diagonal
             D diag_list
    """

    P = u * u.T
    P1 = (P + P.H).real
    P2 = (P - P.H).imag
    P1[:] /= 2
    P2[:] /= 2
    T, D1, D2 = simultaneous_diagonalize(P1, P2)
    return T.H, [(D1[i] + D2[i] * 1j) for i in range(4)]

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
    GateBuilder.setCargs(low)
    GateBuilder.setTargs(high - 1)
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

    g0g1d = g0 * g1.H
    value, u = np.linalg.eig(g0g1d)


    # L = np.mat(u, dtype=complex)
    angl = []
    for v in value:
        angl.append(np.exp(1j * np.angle(v) / 2))
    d = np.mat(np.diag(angl))
    v = d * u.H * g1

    gates = NKS(low, high - 1, u)
    gates1 = uniformlyRz(low, high, angl)
    gates2 = NKS(low, high - 1, v)

    gates.extend(gates1)
    gates.extend(gates2)

    inner_circuit = Circuit(3)
    inner_circuit.set_flush_gates(gates1)
    inner_circuit.print_infomation()

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

def deal(angle):
    for i in range(len(angle)):
        for j in range(i):
            aa = 0
            for k in range(4):
                aa += angle[i][k, 0] * angle[j][k, 0].conj()
            angle[i] -= aa * angle[j]
        aa = 0
        for j in range(4):
            aa += pow(abs(angle[i][j, 0]), 2)
        aa = np.sqrt(aa)
        angle[i] = angle[i] / aa

def NKS_Optional(low, high, U : np.matrix):
    # parameter cal http://arxiv.org/abs/quant-ph/0307190v1
    # Up = (Y⊗Y)U^T(Y⊗Y)
    Uplist = []
    for i in range(4):
        fhi = 1 if (i == 0 or i == 4) else -1
        plist = []
        for j in range(4):
            fhj = 1 if (j == 0 or j == 4) else -1
            plist.append(U[4 - i, 4 - j] * fhi * fhj)
        Uplist.append(plist)
    Up = np.mat(
        [
            Uplist
        ]
    )
    UUp = U * Up
    values = np.linalg.eigvals(UUp)
    Slist = []
    nS = 0
    for value in values:
        Si = np.angle(value) / 2
        Slist.append(Si)
        nS += Si
    nS = np.round(nS / np.pi)
    Slist.sort()
    for i in range(len(Slist) - 1, len(Slist) - nS - 1, -1):
        Slist[i] = Slist[i] - np.pi
    Slist.sort()
    thx = (Slist[0] + Slist[1]) / 2
    thy = (Slist[0] + Slist[2]) / 2
    thz = (Slist[1] + Slist[2]) / 2

    sqrt2 = 1 / np.sqrt(2)
    M = np.mat(
        [
            [1 * sqrt2, 1j * sqrt2, 0, 0],
            [0, 0, 1j * sqrt2, 1 * sqrt2],
            [0, 0, 1j * sqrt2, -1 * sqrt2],
            [1 * sqrt2, -1j * sqrt2, 0, 0]
        ]
    )

def NKS_2qubit(low, high, U : np.matrix):
    # TWO qubit decomposition https://arxiv.org/abs/0806.4015v1
    # 注意其定义的Rz与QuICT中的Rz差一个全局相位和一个负号,Ry也相差一个负号
    global EPS, EPS_bit

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
    value, u = linalg.eig(M2)
    u = np.mat(u)

    angle = []
    for i in range(len(value)):
        angle.append(u[:, i])

    deal(angle)

    # 4 compute D^(1/2) Kp
    P = np.eye(4, dtype=complex)
    for i in range(4):
        P[:, i] = angle[i].T
    P = np.mat(P)
    det = np.linalg.det(P)
    for i in range(4):
        for j in range(4):
            if abs(abs(P[i, j])) > EPS:
                print(i, j, P[i, j], P[i, j].imag * P[i, j].imag / (P[i, j].imag * P[i, j].imag + P[i, j].real * P[i, j].real))

    Dd_2 = []
    Dd2 = []
    for i in range(4):
        the_angle = np.angle(value[i]) / 2
        Dd_2.append(np.exp(-the_angle * 1j))
        Dd2.append(np.exp(the_angle * 1j))
    if np.linalg.det(P).real < 0:
        P[:, 0] = -P[:, 0]
    Dd_2 = np.diag(Dd_2)
    Dd2 = np.diag(Dd2)
    if np.linalg.det(Dd2).real < 0:
        Dd2[:, 0] = -Dd2[:, 0]
    if np.linalg.det(Dd_2).real < 0:
        Dd_2[:, 0] = -Dd_2[:, 0]

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

    # thx = (angle[0][0] + angle[2][0])
    # thy = (angle[1][0] + angle[2][0])
    # thz = (angle[0][0] + angle[1][0])
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
    # recursion

    gates = Decomposition_Recursion(low, high, k_p)


    gates1 = uniformlyRx(low, high, angles_value)
    gates2 = Decomposition_Recursion(low, high, p.H)

    gates.extend(gates1)
    gates.extend(gates2)

    inner_circuit = Circuit(3)
    # inner_circuit.set_flush_gates(gates2)
    # unitary = SyntheticalUnitary.run(inner_circuit)

    return gates

class Cartan_decomposition(Synthesis):
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
        if np.any(U * U.H - np.identity(n) > 1e-6):
            raise Exception("给出的不是合法的量子电路酉矩阵")
        detU = np.linalg.det(U)
        if abs(abs(detU) - 1) > 1e-6:
            raise Exception("给出的不是合法的量子电路酉矩阵")
        circuit = Circuit(high)
        gates = NKS(0, high, U)
        circuit.set_flush_gates(gates)
        return circuit

if __name__ == '__main__':
    Q = np.mat(ortho_group.rvs(dim=4))
    Q = Q.reshape(4, 4)
    diag1 = []
    diag2 = []
    for i in range(4):
        # ran = random.random()
        ran = np.sqrt(2) / 2
        diag1.append(ran)
        diag2.append(np.sqrt(1 - ran * ran))

    d1 = np.diag(diag1)
    d2 = np.diag(diag2)
    A = Q.T * d1 * Q
    B = Q.T * d2 * Q
    U = A + B[:] * 1j


    circuit = Cartan_decomposition.run(U)
    new_unitary = SyntheticalUnitary.run(circuit)

