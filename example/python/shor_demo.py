#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

from QuICT.algorithm import (
    BEAShorFactor,
    HRSShorFactor
)
import datetime

N = int(input("Input the number to be factored: "))

starttime = datetime.datetime.now()
#long running
a = BEAShorFactor.run(N)
endtime = datetime.datetime.now()
print("the found factor is", a)
print("the total running time is",  (endtime - starttime).seconds)