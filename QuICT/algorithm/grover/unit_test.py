#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

from QuICT.algorithm import GroverWithPriorKnowledge
from QuICT.core import *

def main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)

def test_1():
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
