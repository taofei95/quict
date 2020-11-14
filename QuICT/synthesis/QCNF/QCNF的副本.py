#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 11:46 上午
# @Author  : Han Yu
# @File    : QCNF.py

from .._algorithm import Algorithm
from QuICT.models import *
from numpy import floor

k = 0
n = 0
m = 0
ancilla =  0
cnf_list = []
cirlist = []
circuit: Circuit = None

def merge_qubit(qubit_a, qubit_b):
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

    for i in range(c_q - 2):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q(-(i + 1))) , target_q)

    for i in range(c_q - 1):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q[-(i + 1)]) , target_q)

    CCX | merge_qubit(control_q[:2], empty_q[-(c_q - 2)])

    for i in range(c_q - 1):
        CCX | merge_qubit(merge_qubit(control_q[-(i + 1)], empty_q(-(i + 1))) , target_q)


def mct(qubit_list):
    cn = len(qubit_list)
    if cn == 4:
        CCX | qubit_list[:3]
        return
    elif cn == 3:
        CX | qubit_list[:2]
        return
    if cn % 2 == 1:
        k1 = cn // 2 + 1
    else:
        k1 = cn // 2
    k2 = n // 2

    linear_cnt(qubit_list[:k1], qubit_list[k1 : k1 + k2 + 1] ,qubit_list[-1])
    H        | qubit_list[-2]
    S        | qubit_list[-1]
    linear_cnt(qubit_list[k1:k1 + k2 + 1], qubit_list[:k1], qubit_list[-1])
    S_dagger | qubit_list[-1]
    linear_cnt(qubit_list[:k1], qubit_list[k1: k1 + k2 + 1], qubit_list[-1])
    S        | qubit_list[-1]
    linear_cnt(qubit_list[k1:k1 + k2 + 1], qubit_list[:k1], qubit_list[-1])
    H        | qubit_list[-2]
    S_dagger | qubit_list[-1]

class cxcir:
    def __init__(self):
        self.tar = 0
        self.con = []

def Clause(id, plz):
    global k, cnf_list, cirlist, circuit

    c = cxcir()
    c.tar = plz
    cnf = cnf_list[id]
    for thing in cnf:
        if thing > 0:
            tc = cxcir()
            tc.connum = 0
            tc.tar = thing
            cirlist.append(tc)
            X | circuit(thing - 1)
        if thing != 0:
            c.con.append(thing)
    X | circuit(plz - 1)
    qubit_list = []
    for con in c.con:
        qubit_list.append(abs(con) - 1)
    qubit_list.append(plz - 1)
    # 辅助比特
    qubit_list.append(n + ancilla)
    qubit_mct = circuit(qubit_list)
    mct(qubit_mct)
    cirlist.append(c)
    for thing in cnf:
        if thing > 0:
            tc = cxcir()
            tc.tar = thing
            cirlist.append(tc)
            X | circuit(thing - 1)

def Clause_l(stid, endid, plz):
    global n, m, ancilla, cnf_list
    freelist = []
    used = [False] * (n + 1)
    used[plz] = True
    for i in range(stid, endid + 1):
        cnf = cnf_list[i]
        for var in cnf:
            used[abs(var)] = True
    for i in range(0, n):
        if not used[i]:
            freelist.append(i)
    gap = endid - stid + 1
    if gap == 1:
        Clause(stid, plz)
    elif gap == 2:
        tc = cxcir()
        tc.con = freelist[:2]
        tc.tar = plz

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(stid, freelist[0])

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(endid, freelist[1])

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(stid, freelist[0])

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(stid, freelist[1])
    else:
        tc = cxcir()
        tc.con = [freelist[gap - 1], freelist[2 * gap - 3]]
        tc.tar = plz

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(endid, freelist[gap - 1])
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(gap - 2, 1, -1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Clause(endid + (i - gap + 1), freelist[i])
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

        tc.con = freelist[:2]
        tc.tar = freelist[gap]

        CCX | circuit(tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1)
        cirlist.append(tc)
        Clause(stid, freelist[0])

        CCX | circuit(tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1)
        cirlist.append(tc)
        Clause(stid + 1, freelist[1])

        CCX | circuit(tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1)
        cirlist.append(tc)
        Clause(stid, freelist[0])

        CCX | circuit(tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1)
        cirlist.append(tc)

        for i in range(2, gap - 1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Clause(endid + (i - gap + 1), freelist[i])
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

        tc.con = [freelist[gap - 1], freelist[2 * gap - 3]]
        tc.tar = plz
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(endid, freelist[gap - 1])
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(gap - 2, 1, -1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Clause(endid + (i - gap + 1), freelist[i])
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

        tc.con = [freelist[0], freelist[1]]
        tc.tar = freelist[gap]

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(stid, freelist[0])

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(stid + 1, freelist[1])

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Clause(stid, freelist[0])

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(2, gap - 1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 2]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Clause(endid + (i - gap + 1), freelist[i])
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

def Cnf_c(stid, endid, d, plz, siz):
    global m, cnf_list, ancilla
    if d == -1:
        d = 0
        nowa = 1
        temm = siz
        if temm == 1:
            d = 0
        else:
            d += 1
            while (temm - 1) // ancilla >= nowa:
                d += 1
                nowa = nowa * ancilla
        Cnf_c(stid, endid, d, plz, siz)
        return
    elif d == 0:
        Clause_l(stid, endid, plz)
    else:
        gap = min(siz, (ancilla + 1) // 2)
        Cnf_half(stid, endid, d, plz, siz, gap)

def Cnf_half(stid, endid, d, plz, siz, gap):
    freelist = [0] * (m + 1)
    ancist = n + 1
    for i in range(ancilla - 1):
        freelist[i] = ancist + i if (ancist + i < plz) else ancist + i + 1
    csiz = int(floor((endid-stid+1.0) / gap))
    block = int(floor(1.0 * siz / gap))
    if gap == 1:
        Cnf_c(stid, endid, d - 1, plz, siz)
    elif gap == 2:
        tc = cxcir()
        tc.con = freelist[:2]
        tc.tar = plz
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid, stid + csiz - 1, d - 1, freelist[0], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid + csiz, endid, d - 1, freelist[0], siz - block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid, stid + csiz - 1, d - 1, freelist[0], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid + csiz, endid, d - 1, freelist[1], siz - block)
    else:
        tc = cxcir()
        tc.con = [freelist[gap - 1], freelist[2 * gap - 3]]
        tc.tar = plz
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid + csiz * (gap - 1), endid, d - 1, freelist[gap - 1], siz - block * (gap - 1))

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(gap - 2, 1, -1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

            Cnf_c(stid + i * csiz, stid + (i + 1) * csiz - 1, d - 1, freelist[i], block)
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

        tc.con = freelist[:2]
        tc.tar = freelist[gap]
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid, stid + csiz - 1, d - 1, freelist[1], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid + csiz, stid + 2 * csiz - 1, d - 1, freelist[0], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid, stid + csiz - 1, d - 1, freelist[1], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(2, gap - 1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Cnf_c(stid +  i * csiz, stid + (i + 1) * csiz - 1, d - 1, freelist[i], block)

            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

        tc.con = [freelist[gap - 1], freelist[2 * gap - 3]]
        tc.tar = plz
        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid +  csiz * (gap - 1), endid, d - 1, freelist[gap - 1], siz - block * (gap -  1))

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(gap - 2, 1, -1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]

            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Cnf_c(stid + i * csiz, stid + (i + 1) * csiz - 1, d - 1, freelist[i], block)

            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

        tc.con = freelist[:2]
        tc.tar = freelist[gap]

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid, stid + csiz - 1, d - 1, freelist[1], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid + csiz, stid + 2 * csiz - 1, d - 1,  freelist[0], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)
        Cnf_c(stid, stid + csiz - 1, d - 1, freelist[1], block)

        CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
        cirlist.append(tc)

        for i in range(2, gap - 1):
            tc.con = [freelist[i], freelist[i + gap - 2]]
            tc.tar = freelist[i + gap - 1]
            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)
            Cnf_c(stid + i * csiz, stid + (i + 1) * csiz - 1, d - 1, freelist[i], block)

            CCX | circuit([tc.con[0] - 1, tc.con[1] - 1, tc.tar - 1])
            cirlist.append(tc)

def solve():
    global n, m, ancilla, cnf_list
    global k
    k = 0
    for cnf in cnf_list:
        k = max(k, len(cnf))
    plz = ancilla + n

    temg = max(n // (k + 2), 1)
    siz = (m - 1) // temg + 1
    if siz <= ancilla:
        ancilla = siz + 1
    tsiz = (siz - 1) // (ancilla - 1) + 1
    teml = (ancilla + 1) // 2
    d = 0
    bsiz = siz % (ancilla - 1)
    if bsiz == 0:
        bsiz = ancilla - 1
    tsiz -= 1
    while tsiz  > 0:
        tsiz //= teml
        d += 1
    if siz == 1:
        Cnf_c(0, m - 1, 0, plz, siz)
        return

    tsiz = (siz - 1) // (ancilla - 1) + 1
    d2 = 0
    tsiz -= 2
    while tsiz > 0:
        tsiz //= teml
        d2 += 1
    if siz == 1:
        Cnf_c(0, m - 1, 0, plz, siz)
        return

    tsiz = (siz - 1) // (ancilla - 1) + 1
    freelist = [0] * (m + 1)
    ancist = n + 1
    for i in range(ancilla - 1):
        freelist[i] = (ancist + i) if ancist + i < plz else ancist + i + 1

    for i in range(bsiz):
        Cnf_c(i * tsiz, (i + 1) * tsiz - 1, d, i + n + 1, tsiz)

    for i in range(bsiz, ancilla - 1):
        Cnf_c(i * tsiz - (i - bsiz), (i + 1) * tsiz - 2 - i + bsiz, d2, n + i + 1, tsiz - 1)

    tc = cxcir()
    tc.tar = plz
    tc.con = freelist[:ancilla - 1]

    qubit_list = []
    for thing in freelist:
        qubit_list.append(thing - 1)
    qubit_list.append(plz - 1)
    qubit_list.append(n + ancilla)
    qubit_mct = circuit(qubit_list)
    mct(qubit_mct)
    cirlist.append(tc)

    for i in range(bsiz, ancilla - 1):
        Cnf_c(i * tsiz - (i - bsiz), (i + 1) * tsiz - 2 - i + bsiz, d2, n + i + 1, tsiz - 1)
    for i in range(bsiz):
        Cnf_c(i  * tsiz, (i + 1) * tsiz - 1, d, i + n + 1, tsiz)

class QCNF(Algorithm):
    @staticmethod
    def __run__(_n, _m, _ancilla, _cnf_list):
        global n, m, ancilla, cnf_list, circuit
        """
        :param n: 变量数
        :param m: 子句数目
        :param ancilla: 辅助比特数，要求>=4
        :param cnf_list: cnf列表，包含m个list，每个list包含-n～n（不包括0）的整数，表示第x个变量或者取反
        :return: 返回结果
        """
        if _ancilla < 4:
            raise Exception("辅助比特数至少为4")
        n = _n
        m = _m
        ancilla = _ancilla - 1
        cnf_list = _cnf_list
        circuit = Circuit(n + ancilla + 1)
        return solve()
