#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

from QuICT.algorithm import HRS_order_finding

#N = int(input("Input the modulo N: "))
#a = int(input("Input the element wanting the order: "))

order = HRS_order_finding.run(7,33,'demo')
if order != 0:
    print("HRS_order_finding found order", order)