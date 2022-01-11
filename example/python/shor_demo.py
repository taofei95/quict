#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/2 9:58
# @Author  : Zhu Qinlin
# @File    : shor_demo.py

from QuICT.algorithm import HRSShorFactor, BEAShorFactor
#TODO: import /QuICT from /example/python
# import QuICT
# print(QuICT.__file__)

N = int(input("[HRS]Input the number to be factored: "))
a = HRSShorFactor.run(N,5,'demo')
print("HRSShor found factor", a)

N = int(input("[BEA]Input the number to be factored: "))
a = BEAShorFactor.run(N,5,'demo')
print("BEAShor found factor", a)