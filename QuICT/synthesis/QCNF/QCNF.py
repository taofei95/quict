#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 11:46 上午
# @Author  : Han Yu
# @File    : QCNF.py

from .._synthesis import Synthesis
from QuICT.synthesis.MCT import MCT_one_aux
from QuICT.models import *
import copy

# 变量数n
n = 0
# 子句数m
m = 0
# 子句最长数k
k = 0
# CNF子句列表
cnf_list = []
# 辅助比特数
ancilla = 0
# 电路
circuit = None
# 返回电路
result = None

def var_assignment(circuit, assignment):
    if assignment is None:
        return
    i = 0
    for ass in assignment:
        if ass == 1:
            X | circuit(i)
        i += 1

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

def Clause(id, plz):
    """
    将序号为id的子句作用到plz位上
    :param id:  子句序号
    :param plz: 作用位
    """
    global k, cnf_list, circuit

    print("Clause", id, plz)

    qubit_list = []
    cnf = cnf_list[id]
    for thing in cnf:
        if thing > 0:
            X | circuit(thing - 1)
        if thing != 0:
            qubit_list.append(abs(thing) - 1)
    X | circuit(plz)

    qubit_list.append(plz)
    qubit_list.append(n + ancilla)
    qubit_mct = circuit(qubit_list)
    MCT_one_aux | qubit_mct
    for thing in cnf:
        if thing > 0:
            X | circuit(thing - 1)


def tool_struct_one(stid, plz):
    """
    将子句stid作用到plz上
    :param stid: 子句id
    :param plz:  作用位
    """
    global circuit
    Clause(stid, plz)

def tool_struct_two(stid, endid, plz, freelist, gap):
    """
    endid - stid = 2时，将子句[stid, endid)的和作用到plz上
    :param stid:     开始子句id
    :param endid:    结束子句id
    :param plz:      作用位
    :param freelist: 脏辅助比特列表
    :param gap:      一次可以合成多少
    :return:
    """
    global circuit
    rec_size = (endid - stid) // gap

    CCX | circuit([freelist[0], freelist[1], plz])
    Cnf_half(stid, stid + rec_size, freelist[0])

    CCX | circuit([freelist[0], freelist[1], plz])
    Cnf_half(stid + rec_size, endid, freelist[1])

    CCX | circuit([freelist[0], freelist[1], plz])
    Cnf_half(stid, stid + rec_size, freelist[0])

    CCX | circuit([freelist[0], freelist[1], plz])
    Cnf_half(stid + rec_size, endid, freelist[1])

def tool_struct_more(stid, endid, plz, freelist, gap):
    """
    endid - stid > 2时，将子句[stid, endid)的和作用到plz上
    :param stid:     开始子句id
    :param endid:    结束子句id
    :param plz:      作用位
    :param freelist: 脏辅助比特列表
    :param gap:      一次可以合成多少
    :return:
    """
    global circuit
    rec_size = (endid - stid) // gap

    CCX | circuit([freelist[gap - 1], freelist[2 * gap - 3], plz])
    Cnf_half(stid + rec_size * (gap - 1), endid, freelist[gap - 1])
    CCX | circuit([freelist[gap - 1], freelist[2 * gap - 3], plz])

    for i in range(gap - 2, 1, -1):
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])
        Cnf_half(stid + i * rec_size, stid + (i + 1) * rec_size, freelist[i])
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])

    CCX | circuit([freelist[0], freelist[1], freelist[gap]])
    Cnf_half(stid, stid + rec_size, freelist[1])
    CCX | circuit([freelist[0], freelist[1], freelist[gap]])

    Cnf_half(stid + rec_size, stid + 2 * rec_size, freelist[0])

    CCX | circuit([freelist[0], freelist[1], freelist[gap]])
    Cnf_half(stid, stid + rec_size, freelist[1])
    CCX | circuit([freelist[0], freelist[1], freelist[gap]])

    for i in range(2, gap - 1):
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])
        Cnf_half(stid + i * rec_size, stid + (i + 1) * rec_size, freelist[i])
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])

    CCX | circuit([freelist[gap - 1], freelist[2 * gap - 3], plz])
    Cnf_half(stid + rec_size * (gap - 1), endid, freelist[gap - 1])
    CCX | circuit([freelist[gap - 1], freelist[2 * gap - 3], plz])

    for i in range(gap - 2, 1, -1):
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])
        Cnf_half(stid + i * rec_size, stid + (i + 1) * rec_size, freelist[i])
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])

    CCX | circuit([freelist[0], freelist[1], freelist[gap]])
    Cnf_half(stid, stid + rec_size, freelist[1])
    CCX | circuit([freelist[0], freelist[1], freelist[gap]])

    Cnf_half(stid + rec_size, stid + 2 * rec_size, freelist[0])

    CCX | circuit([freelist[0], freelist[1], freelist[gap]])
    Cnf_half(stid, stid + rec_size, freelist[1])
    circuit([freelist[0], freelist[1], freelist[gap]])

    for i in range(2, gap - 1):
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])
        Cnf_half(stid + i * rec_size, stid + (i + 1) * rec_size, freelist[i])
        CCX | circuit([freelist[i], freelist[i + gap - 2], freelist[i + gap - 1]])

def Clause_l(stid, endid, plz):
    """
    用n个qubit将[stid, endid)的子句的求和异或到plz位上
    :param stid:  开始子句
    :param endid: 结束子句
    :param plz:   目标位
    """
    global n, m, ancilla, cnf_list, circuit
    freelist = []
    used = [False] * (n + ancilla + 2)
    used[plz] = True
    for i in range(stid, endid):
        cnf = cnf_list[i]
        for var in cnf:
            used[abs(var)] = True
    for i in range(0, n):
        if not used[i]:
            freelist.append(i)
    gap = endid - stid

    #  print("Clause_l", stid, endid, plz)

    if gap == 1:
        Clause(stid, plz)
    elif gap == 2:
        print("Clause_l", stid, endid, plz)
        tool_struct_two(stid, endid, plz, freelist, gap)
    else:
        print("Clause_l", stid, endid, plz)
        tool_struct_more(stid, endid, plz, freelist, gap)

def Cnf_half(stid, endid, plz):
    """
    用辅助比特将[stid, endid)的子句的求和异或到plz位上
    :param stid:  开始子句
    :param endid: 结束子句
    :param plz:   目标位
    """
    item_size = endid - stid
    if item_size == 0:
        return
    item_number = max(n // (k + 2), 1)
    if item_size <= item_number:
        Clause_l(stid, endid, plz)
        return
    gap = min((ancilla + 1) // 2, item_size)

    freelist = [0] * (m + 1)
    anc_index = n
    for i in range(ancilla - 1):
        freelist[i] = anc_index + i if (anc_index + i < plz) else anc_index + i + 1
    if gap == 1:
        Clause(stid, plz)
    elif gap == 2:
        tool_struct_two(stid, endid, plz, freelist, gap)
    else:
        tool_struct_more(stid, endid, plz, freelist, gap)

def solve():
    global n, m, ancilla, cnf_list, circuit, k, result


    # 获取CNF最长长度
    k = 0
    for cnf in cnf_list:
        k = max(k, len(cnf))

    # 用n个比特一个可以合并n / (k + 2)项的k-CNF，item_number表示最后一次合并时最多可合并的项数
    # 此时这n个比特作为结构上的辅助比特，子句在原本的辅助比特上
    item_number = max(n // (k + 2), 1)

    # 将m个子句分为item_number组，item_size为每组需要合并的数量（需要上取整）
    item_size = (m - 1) // item_number + 1

    # 作用位序号，为电路的倒数第一位，倒数第二位为多控制toffoli门的辅助，这里我们把最后两位反过来
    # 于是在代码中作用位序号，为电路的倒数第二位，倒数第一位为多控制toffoli门的辅助
    # 辅助比特数至多需要item_size + 2
    result = Circuit(n + ancilla + 1)

    # 辅助比特数至多需要item_size + 2
    if item_size <= ancilla - 2:
        ancilla = item_size + 2

    circuit = result([i for i in range(n + ancilla - 1)])
    X | circuit(3)
    X | circuit(4)
    X | circuit(6)
    X | circuit(7)
    X | circuit(9)
    circuit.append(result[result.circuit_length() - 1])
    circuit.append(result[n + ancilla - 1])
    plz = len(circuit) - 2

    # 如果子式数量不多于n // (k + 2)，那么直接合并到plz位就好了
    if item_size == 1:
        Clause_l(0, m, plz)
        return result

    # 每个辅助比特需要承担的项数
    item_mod_ancilla = item_size % (ancilla - 1)
    if item_mod_ancilla == 0:
        item_mod_ancilla = (ancilla - 1)

    gate_length = len(result.gates)

    # 将最后合并前子式求和到辅助比特，项尽量均匀分布（前后最多差1）
    item_per_ancilla = (item_size - 1) // (ancilla - 1) + 1
    print(item_size, ancilla, item_per_ancilla, item_mod_ancilla, ancilla - 1 - item_mod_ancilla)
    for i in range(item_mod_ancilla):
        Cnf_half(i * item_per_ancilla, (i + 1) * item_per_ancilla, n + i)
    total = item_mod_ancilla * item_per_ancilla
    if item_mod_ancilla != 0:
        item_per_ancilla -= 1
    for i in range(ancilla - 1 - item_mod_ancilla):
        Cnf_half(total + i * item_per_ancilla, total + (i + 1) * item_per_ancilla, n + i + item_mod_ancilla)

    gates = copy.deepcopy(result.gates[gate_length:])

    # 将子式子求和到作用位上
    qubit_list = []
    for thing in range(ancilla - 1 if item_per_ancilla > 0 else item_mod_ancilla):
        qubit_list.append(n + thing)
    qubit_list.append(plz)
    qubit_list.append(len(circuit) - 1)
    qubit_mct = circuit(qubit_list)
    MCT_one_aux | qubit_mct

    # 还原辅助比特
    GateBuilder.reflect_apply_gates(gates, result)

    return result

class QCNFModel(Synthesis):

    def __call__(self, _n, _m, _ancilla, _cnf_list):
        self.pargs = [_n, _m, _ancilla, _cnf_list]
        if _ancilla < 4:
            raise Exception("辅助比特数至少为4")
        return self

    def build_gate(self):
        global n, m, ancilla, cnf_list
        n = self.pargs[0]
        m = self.pargs[1]
        ancilla = self.pargs[2] - 1
        cnf_list = self.pargs[3]
        return solve()

QCNF = QCNFModel()
