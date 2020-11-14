#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/4/21 7:28 下午
# @Author  : Han Yu
# @File    : _SK_decomposition.py

import numpy as np
from .._synthesis import Synthesis
from QuICT.models import *

eps0 = 0.14
l0 = 16
c_approx = 1 / np.sqrt(2)

generator_matrix = []
generator_matrix_list = []

matrixs = None
gates = None

best_l = []
best_u = None
best_eps = 1

now_l = []
now_u = None

def choose_matrix():
    global matrixs, gates
    matrixs = []
    gates = []

    GateBuilder.setGateType(GateType.H)
    GateBuilder.setTargs(0)
    gates.append(GateBuilder.getGate())
    matrixs.append(np.asmatrix(H.matrix.reshape(2, 2)))

    GateBuilder.setGateType(GateType.U1)
    GateBuilder.setTargs(0)
    GateBuilder.setPargs(np.pi / 8)
    gates.append(GateBuilder.getGate())
    matrixs.append(np.asmatrix(gates[1].matrix.reshape(2, 2)))

    GateBuilder.setPargs(-np.pi / 8)
    gates.append(GateBuilder.getGate())
    matrixs.append(np.asmatrix(gates[2].matrix.reshape(2, 2)))

def best_match(U, l):
    global matrixs, gates, now_u, now_l, best_eps, best_u, best_l
    try_eps = np.linalg.norm(now_u - U)
    if try_eps < best_eps:
        best_eps = try_eps
        best_u = now_u
        best_l = now_l.copy()
    if l + 1 == l0:
        return
    for i in range(len(matrixs)):
        now_u = now_u * matrixs[i]
        now_l.append(gates[i])
        best_match(U, l + 1)
        now_l.pop()
        now_u = now_u * matrixs[i].H

def Basic_Approxiamtion(U):
    global best_u, now_u
    best_u = np.identity(U.shape[0])
    now_u = np.identity(U.shape[0])
    best_match(U, 0)
    print(best_l)
    return best_l, best_u

def GC_Approx_Decompose(U):
    # H = -ilog(U)
    value, u = np.linalg.eig(U)
    u = np.mat(u)
    for i in range(len(value)):
        value[i] = -1j * np.log(value[i])
    HH = u * np.diag(value) * u.H
    d = np.shape(U)[0]
    G = np.diag([-(d - 1) / 2 + i for i in range(d)])
    F = np.mat(np.zeros(d, d))
    for j in range(d):
        for k in range(d):
            if j == k:
                F[j, k] = 0
            else:
                F[j, k] = 1j * HH[j, k] / (G[k, k] - G[j, j])
    f = abs(np.linalg.det(F))
    g = (d - 1) / 2
    scale = np.sqrt(f * g)
    F[:, :] = np.power(scale / f, 1 / d) * F[:, :]
    G[:, :] = np.power(scale / g, 1 / d) * G[:, :]

    value, u = np.linalg.eig(F)
    u = np.mat(u)
    for i in range(len(value)):
        value[i] = 1j * np.exp(value[i])
    V = u * np.diag(value) * u.H

    value, u = np.linalg.eig(G)
    u = np.mat(u)
    for i in range(len(value)):
        value[i] = 1j * np.exp(value[i])
    W = u * np.diag(value) * u.H
    return V, W

def Solovay_Kitaev(U, n):
    if n == 0:
        return Basic_Approxiamtion(U)
    U_nd1l, U_nd1 = Solovay_Kitaev(U, n - 1)
    V, W = GC_Approx_Decompose(U * U_nd1)
    V_nd1l, V_nd1 = Solovay_Kitaev(V, n - 1)
    W_nd1l, W_nd1 = Solovay_Kitaev(W, n - 1)
    Unl = V_nd1l
    Unl.extend(V_nd1l)
    Unl.extend(W_nd1l)
    Unl.extend(list(reversed(V_nd1l)))
    Unl.extend(list(reversed(W_nd1l)))
    Unl.extend(U_nd1l)
    Un = V_nd1 * W_nd1 * V_nd1.H * W_nd1.H * U_nd1
    return Unl, Un

class SK_decompostion_model(Synthesis):
    def __call__(self, U: np.matrix, eps = 0.14):
        self.pargs = [U, eps]
        return self

    def build_gate(self):
        """
        SK分解
        :param U: 待分解酉矩阵
        :param eps: 精度
        :return: 返回电路门的数组
        """
        U = self.pargs[0]
        eps = self.pargs[1]
        if matrixs is None:
            choose_matrix()
        global best_u, now_u
        n = int(np.ceil(np.log(np.log(1 / eps / c_approx / c_approx)
                               / np.log(1 / eps0 / c_approx / c_approx)) / np.log(3 / 2)))
        best_u = np.identity(U.shape[0])
        now_u = np.identity(U.shape[0])
        r_gates, _ = Solovay_Kitaev(U, n)
        circuit_back = Circuit(1)
        circuit_back.set_flush_gates(r_gates)
        return circuit_back, best_u

SK_decompostion = SK_decompostion_model()
