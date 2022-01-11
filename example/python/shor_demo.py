#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

from QuICT.algorithm import (
    BEAShorFactor,
    HRSShorFactor
)

N = int(input("Input the number to be factored: "))

a = BEAShorFactor.run(N,5,'demo')

print("BEAShor found factor", a)

#a = HRSShorFactor.run(N,5,'demo')

#print("HRSShor found factor", a)