#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

from QuICT.algorithm import (
    BEAShorFactor,
    BEA_order_finding,
    HRSShorFactor,
    HRS_order_finding
)

#N = int(input("Input the modulo N: "))
#a = int(input("Input the element wanting the order: "))

order = HRS_order_finding.run(4,21,'demo')

print("HRS_order_fingding found order", order)

order = BEA_order_finding.run(4,21,'demo')

print("BEA_order_fingding found order", order)


#a = HRSShorFactor.run(N,5,'demo')

#print("HRSShor found factor", a)