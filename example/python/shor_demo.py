#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/2 9:58
# @Author  : Zhu Qinlin
# @File    : shor_demo.py

from QuICT.algorithm import HRSShorFactor

N = int(input("Input the number to be factored: "))

a = HRSShorFactor.run(N,5,'demo')

print("HRSShor found factor", a)

#a = HRSShorFactor.run(N,5,'demo')

#print("HRSShor found factor", a)