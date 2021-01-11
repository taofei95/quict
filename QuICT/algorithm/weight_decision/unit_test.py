#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

import random

from QuICT.algorithm import WeightDecision
from QuICT.core import *

def randomList(_rand):
    n = len(_rand) - 2
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]

def deutsch_jozsa_main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)

def test_1():
    for test_number in range(3, 5):
        for k in range(1, 8):
            for l in range(k + 1, 8):
                for T in range(l, 20):
                    for _ in range(10):
                        ans = random.randint(0, 1)
                        if ans == 0:
                            final = k
                        else:
                            final = l
                        test = [1] * final
                        for i in range(final, 1 << int(np.ceil(np.log2(T + 2)))):
                            test.append(0)
                        randomList(test)
                        ans = WeightDecision.run(test, T, k, l, deutsch_jozsa_main_oracle)
                        print(test, T, k, l)
                        print(final, ans)
                        assert final == ans

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
