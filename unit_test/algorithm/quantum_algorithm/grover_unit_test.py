#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:58
# @Author  : Zhu Qinlin
# @File    : unit_test.py

import pytest

from QuICT.algorithm.quantum_algorithm.grover import (
    Grover,
    PartialGrover,
    GroverWithPriorKnowledge,
)
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def main_oracle(n, f):
    assert len(f) == 1, "only 1 target support for this oracle"
    assert f[0] >= 0 and f[0] < (1 << n)

    result_q = [n]
    cgate = CompositeGate()
    target_binary = bin(f[0])[2:].rjust(n, "0")
    with cgate:
        # |-> in result_q
        X & result_q[0]
        H & result_q[0]
        # prepare for MCT
        for i in range(n):
            if target_binary[i] == "0":
                X & i
    MCTOneAux.execute(n + 2) | cgate
    # un-compute
    with cgate:
        for i in range(n):
            if target_binary[i] == "0":
                X & i
        H & result_q[0]
        X & result_q[0]
    return 2, cgate


def test_grover_on_ConstantStateVectorSimulator():
    for n in range(3, 9):
        error = 0
        N = 2 ** n
        for target in range(0, N):
            f = [target]
            k, oracle = main_oracle(n, f)
            result = Grover.run(n, k, oracle, ConstantStateVectorSimulator())
            if target != result:
                error += 1
                print("For n = %d, target = %d, found = %d" % (n, target, result))
        error_rate = error / N
        print(
            "for n = %d, %d errors in %d tests, error rate = %f"
            % (n, error, N, error_rate)
        )
        if error_rate > 0.15:
            assert 0
    assert 1


def test_partial_grover_on_ConstantStateVectorSimulator():
    n_block = 3
    for n in range(5, 9):
        print("run with n = ", n)
        error = 0
        N = 2 ** n
        for target in range(0, N):
            f = [target]
            k, oracle = main_oracle(n, f)
            result = PartialGrover.run(
                n, n_block, k, oracle, ConstantStateVectorSimulator()
            )
            if (target >> (n - k)) != (result >> (n - k)):
                error += 1
        error_rate = error / N
        print(
            "for n = %d, %d errors in %d tests, error rate = %f"
            % (n, error, N, error / N)
        )
        if error_rate > 0.15:
            assert 0
    assert 1


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
