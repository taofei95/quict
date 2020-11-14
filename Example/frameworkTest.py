#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/17 8:02 下午
# @Author  : Han Yu
# @File    : frameworkTest.py

from QuICT.models import *
import sys
'''
对于不带参数的门，直接用｜语法作用在Qubit/Qureg上即可，默认控制位在前，作用位在后，例如
'''
circuit = Circuit(3)    # 构建一个3位的量子电路
X | circuit(0)          # 在电路的0位上作用一个X门
qureg = circuit([0, 2])	#	获取电路的0位和2位组成的Qureg
CX | qureg              # 在qureg上作用CNOT门，控制位为0，作用位为2
# 当一个单比特门作用于一个Qureg上时，默认会作用在所有Qubit上
Measure | qureg         # 在电路的0位和2位上各作用一个测量门
'''
对于带参数的门，用｜语法作用在Qubit/Qureg上的同时还需要指定参数，参数用小括号给出
作用位的个数会通过参数的个数自动计算，如果不符合约束，会进行报错
'''
Perm([0, 1, 3, 2]) | qureg           # 将10和11的振幅进行置换
Custom([0, 1, 1, 0]) | circuit(0)    # 自定义一个X门，作用在电路0位上

circuit.print_infomation()  # 输出电路信息
circuit.flush()         # 执行以上门
print(qureg)
# 输出qureg对应的测量结果，在前的Qubit为低位，即输出('0'的测量结果 + '2'的测量结果 * 2)
circuit = Circuit(3)
