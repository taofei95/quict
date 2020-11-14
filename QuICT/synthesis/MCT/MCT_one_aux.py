#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 11:37 上午
# @Author  : Han Yu
# @File    : MCT_one_aux.py

from .._synthesis import Synthesis
from ..MCT.MCT_Linear_Simulation import MCT_Linear_Simulation
from QuICT.models import *


def merge_qubit(qubit_a, qubit_b):
    """
    合并两个qureg列表为一个
    :param qubit_a: 列表a
    :param qubit_b: 列表b
    :return: 合并后的列表
    """
    qureg = []
    if isinstance(qubit_a, Qubit):
        qureg.append(qubit_a)
    else:
        for qubit in qubit_a:
            qureg.append(qubit)
    if isinstance(qubit_b, Qubit):
        qureg.append(qubit_b)
    else:
        for qubit in qubit_b:
            qureg.append(qubit)
    return qureg

def linear_cnt(control_q, empty_q, target_q):
    """
    https://arxiv.org/abs/quant-ph/9503016 Lemma 7.2
    If n ≥ 5 and m ∈ {3, . . . , ⌈n/2⌉} then (m+1)-Toffoli gate can be simulated
    by a network consisting of 4(m − 2) toffoli gates
    :param control_q: 控制位列表
    :param empty_q:  空位列表
    :param target_q: 目标位
    """
    c_q = len(control_q)
    if c_q <= 2:
        if c_q == 2:
            CCX | merge_qubit(control_q, target_q)
        elif c_q == 1:
            CX | merge_qubit(control_q, target_q)
        return

    for i in range(c_q - 2):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q[-(i + 1)]) , target_q)

    CCX | merge_qubit(control_q[:2], empty_q[-(c_q - 2)])

    for i in range(c_q - 2 - 1, -1, -1):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q[-(i + 1)]) , target_q)

    for i in range(1, c_q - 2):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q[-(i + 1)]) , target_q)

    CCX | merge_qubit(control_q[:2], empty_q[-(c_q - 2)])

    for i in range(c_q - 2 - 1, 0, -1):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q[-(i + 1)]) , target_q)

def solve(n):
    """
    He Y, Luo M X, Zhang E, et al.
    Decompositions of n-qubit Toffoli gates with linear circuit complexity[J].
    International Journal of Theoretical Physics, 2017, 56(7): 2350-2361.
    用一个辅助比特实现多控制Toffoli门
    :param n: ^n Toffoli门
    :return 返回对应电路
    """
    qubit_list = Circuit(n + 1)
    if n == 3:
        CCX | qubit_list[:3]
        return qubit_list
    elif n == 2:
        CX | qubit_list[:2]
        return qubit_list
    if n % 2 == 1:
        k1 = n // 2 + 1
    else:
        k1 = n // 2
    k2 = n // 2 - 1

    MCT_Linear_Simulation(k1) | qubit_list
    H        | qubit_list[-2]
    S        | qubit_list[-1]
    MCT_Linear_Simulation(k2 + 1) | merge_qubit(merge_qubit(qubit_list[k1:k1 + k2 + 1], qubit_list[:k1]), qubit_list[-1])
    S_dagger | qubit_list[-1]
    MCT_Linear_Simulation(k1) | qubit_list
    S        | qubit_list[-1]
    MCT_Linear_Simulation(k2 + 1) | merge_qubit(merge_qubit(qubit_list[k1:k1 + k2 + 1], qubit_list[:k1]), qubit_list[-1])
    H        | qubit_list[-2]
    S_dagger | qubit_list[-1]

    return qubit_list

class MCT_one_aux_model(Synthesis):
    def build_gate(self):
        n = self.targets - 1
        return solve(n)

MCT_one_aux = MCT_one_aux_model()
