#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:58
# @Author  : Zhu Qinlin
# @File    : unit_test.py

import pytest

from QuICT.algorithm import SimpleGrover, GroverWithPriorKnowledge
from QuICT.core import *

def main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)

def test_SimpleGrover():
    for n in range(3, 9):
        #print("run with n = ",n)
        error = 0
        N =  2**n
        for target in range(0, N):
            f = [0] * N
            f[target] = 1
            #print("For n = %d, target = %d run grover" %(n, target))
            result = SimpleGrover.run(f, n, main_oracle)
            if target != result:
                error += 1
                #print("target = %d, found = %d" %(target, result))
        error_rate = error/N
        if error_rate > 0.2:
            print("for n = %d, %d errors in %d tests, error rate = %f" %(n, error, N, error/N))
            assert 0
    assert 1
            

def test_GroverWithPriorKnowledge():
    for test_number in range(3, 5):
        for i in range(2, 8):
            for T in range(1, 4):
                test = [0] * (1 << test_number)
                test[i] = 1
                prob = [1 / 4, 0, 1 / 4, 1 / 4, 0, 1 / 4, 0, 0]
                for j in range(8, 1 << test_number):
                    prob.append(0)
                p = np.array(prob)
                p /= p.sum()
                GroverWithPriorKnowledge.run(test, 2**test_number, p, T, main_oracle)

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
