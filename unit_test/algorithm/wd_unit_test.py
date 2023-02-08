#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random
import numpy as np

from QuICT.algorithm import WeightDecision
from QuICT.core.gate import *


def randomList(_rand, n):
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]


def test_1():
    k = 6
    l = k + 1
    T = l + 1
    for _ in range(10):
        ans = random.randint(0, 1)
        if ans == 0:
            final = k
        else:
            final = l

        f = [1] * final
        for i in range(final, 1 << int(np.ceil(np.log2(T + 2)))):
            f.append(0)

        np.random.shuffle(f[:T])
        idx = [i for i in range(len(f)) if f[i] == 1]
        flag = False
        perm_gate = PermFx(int(np.ceil(np.log2(T + 2))), idx)
        for _ in range(5):
            ans = WeightDecision.run(T, k, l, perm_gate)
            if final == ans:
                flag = True
                break

        assert flag


if __name__ == '__main__':
    pytest.main()
