#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:58
# @Author  : Zhu Qinlin
# @File    : unit_test.py

import pytest

from QuICT.algorithm.quantum_algorithm.grover import Grover, PartialGrover, GroverWithPriorKnowledge
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def main_oracle(n, f):
    return PermFx(n, f)


def create_simulator():
    return ConstantStateVectorSimulator()


def test_grover_on_ConstantStateVectorSimulator():
    for n in range(3, 9):
        error = 0
        N = 2**n
        for target in range(0, N):
            f = [target]
            result = Grover.run(n, main_oracle(n, f))
            if target != result:
                error += 1
                print("For n = %d, target = %d, found = %d" %
                      (n, target, result))
        error_rate = error / N
        if error_rate > 0.2:
            print("for n = %d, %d errors in %d tests, error rate = %f"
                  % (n, error, N, error_rate))
            assert 0
    assert 1


def test_partial_grover_on_ConstantStateVectorSimulator():
    k = 3
    for n in range(5, 9):
        print("run with n = ", n)
        error = 0
        N = 2**n
        for target in range(0, N):
            f = [target]
            result = PartialGrover.run(n, k, main_oracle(n,f))
            if (target >> (n - k)) != (result >> (n - k)):
                error += 1
        error_rate = error / N
        print("for n = %d, %d errors in %d tests, error rate = %f" %
              (n, error, N, error / N))
        if error_rate > 0.2:
            assert 0
    assert 1


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
