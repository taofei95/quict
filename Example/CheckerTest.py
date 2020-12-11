#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/24 11:18 下午
# @Author  : Han Yu
# @File    : CheckerTest.py

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.optimization import alter_depth_decomposition
from QuICT.checker import ETChecker
from QuICT.models import *
from QuICT.models._gate import GATE_ID


ETChecker.setAlgorithm(alter_depth_decomposition)
ETChecker.setSize(10, 10)
ETChecker.setQubitNumber(6, 8)
ETChecker.setRoundNumber(100)
ETChecker.setTypeList([GATE_ID['X'], GATE_ID['CX'], GATE_ID['CCX'], GATE_ID['ID'], GATE_ID['Swap'],GATE_ID['Perm']])
# ETChecker.setTypeList([GateType.Perm])
ETChecker.run()


'''
circuit = Circuit(6)
# circuit.reset_initial_values()
# Perm([6, 30, 22, 14, 21, 25, 19, 8, 0, 9, 28, 3, 2, 11, 17, 16, 12, 18, 7, 24, 26, 31, 15, 13, 10, 4, 5, 1, 27, 20, 29, 23]) | circuit([4, 7, 0, 5, 3])
X | circuit(5)
result_circuit = ALTER_DEPTH_DECOMPOSITION.run(circuit)
result_circuit.print_infomation()
print(Amplitude.run(result_circuit))

a = [31, 29, 30, 28, 27, 25, 26, 24, 23, 21, 22, 20, 19, 17, 18, 16, 15, 13, 14, 12, 11, 9, 10, 8, 7, 5, 6, 4, 3, 1, 2, 0]
b = [15, 13, 14, 12, 11, 9, 10, 8, 7, 5, 6, 4, 3, 1, 2, 0, 31, 29, 30, 28, 27, 25, 26, 24, 23, 21, 22, 20, 19, 17, 18, 16]

for i in range(len(a)):
    print(i, b[a[i]])

aa = [31, 23, 15, 7, 27, 19, 11, 3, 29, 21, 13, 5, 25, 17, 9, 1, 30, 22, 14, 6, 26, 18, 10, 2, 28, 20, 12, 4, 24, 16, 8, 0]
bb = [30, 22, 14, 6, 26, 18, 10, 2, 28, 20, 12, 4, 24, 16, 8, 0, 31, 23, 15, 7, 27, 19, 11, 3, 29, 21, 13, 5, 25, 17, 9, 1]
for i in range(len(aa)):
    print(i, bb[aa[i]]) 

'''
'''
circuit = Circuit(3)
X  | circuit(0)
CX | circuit([0, 1])
print(SyntheticalUnitary.run(circuit))
'''