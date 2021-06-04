#!/usr/bin/env python
# -*- coding:utf8 -*-

import pytest

from .gr import *
from QuICT.core import *


def main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)


def test_1():
    for test_number in range(3, 5):
        for i in range(8):
            test = [0] * (1 << test_number)
            test[i] = 1
            ans = StandardGrover.run(test, main_oracle)
            # print("[%2d in %2d]answer:%2d"%(i,1<<test_number,ans))


def test_PartialGrover():
    k = 3
    for n in range(5, 10):
        print("run with n = ", n)
        error = 0
        N = 2**n
        for target in range(0, N):
            f = [0] * N
            f[target] = 1
            result = PartialGrover.run(f, n, k, main_oracle)
            if (target >> (n-k)) != (result >> (n-k)):
                # print("[%10s]targetBlock = %s, foundBlock = %s" %
                #       (bin(target), bin(target >> (n-k)), bin(result >> (n-k))))
                error += 1
        error_rate = error/N
        print("for n = %d, %d errors in %d tests, error rate = %f" %
              (n, error, N, error/N))
        if error_rate > 0.2:
            assert 0
    assert 1


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
