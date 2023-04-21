#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/1 16:58
# @Author  : Zhu Qinlin
# @File    : unit_test.py

import unittest
import math
import random

from QuICT.algorithm.quantum_algorithm.grover import (
    Grover,
    PartialGrover,
)

from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.core.gate.backend import MCTOneAux


def main_oracle(n, f):
    assert len(f) == 1, "only 1 target support for this oracle"
    assert 0 <= f[0] < (1 << n)

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
    MCTOneAux().execute(n + 2) | cgate
    # un-compute
    with cgate:
        for i in range(n):
            if target_binary[i] == "0":
                X & i
        H & result_q[0]
        X & result_q[0]
    return 2, cgate


def unknown_oracle(n):
    assert n >= 3
    cgate = CompositeGate()
    CCZ | cgate([n - 3, n - 2, n - 1])
    return 1, cgate


def unknown_oracle_test(s):
    return s[-3:] == "111"


class TestGrover(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Grover unit test start!")
        cls.simulator = StateVectorSimulator(matrix_aggregation=False)

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Grover unit test finished!")

    def test_grover(self):
        n = 3
        error = 0
        N = 2 ** n
        for target in range(0, N):
            f = [target]
            k, oracle = main_oracle(n, f)
            grover = Grover(simulator=TestGrover.simulator)
            result = grover.run(n, k, oracle)
            if target != result:
                error += 1
                print("For n = %d, target = %d, found = %d" % (n, target, result))
        error_rate = error / N
        print(
            "for n = %d, %d errors in %d tests, error rate = %f"
            % (n, error, N, error_rate)
        )
        if error_rate > 1 / math.sqrt(N):
            assert 0

    def test_grover_with_unknown_amplitude(self):
        n = 3
        error = 0
        k, oracle = unknown_oracle(n)
        grover = Grover(simulator=TestGrover.simulator)
        M = 25
        unknown_oracle_test_int = lambda x: unknown_oracle_test(
            bin(x)[2:].rjust(n, "0")
        )
        for _ in range(M):
            solution = grover.run(
                n=n,
                n_ancilla=k,
                oracle=oracle,
                n_solution=None,
                check_solution=unknown_oracle_test_int,
            )
            if solution is None or not unknown_oracle_test_int(solution):
                error += 1
        error_rate = error / M
        print(
            "for n = %d, %d errors in %d tests, error rate = %f\n"
            % (n, error, M, error_rate)
        )

        assert 1

    def test_grover_with_unknown_amplitude_BHMT(self):
        n = 3
        error = 0
        N = 2 ** n
        k, oracle = unknown_oracle(n)
        grover = Grover(simulator=TestGrover.simulator)
        M = 25
        for _ in range(M):
            # BHMT algorithm
            c = math.sqrt(2)
            n_iter = 2
            n_iter_max = 4 * math.ceil(math.sqrt(N))
            n_orac = 0
            solution_found = False
            while n_orac < n_iter_max:
                s_trial_1 = bin(random.randrange(0, N))[2:].rjust(n, "0")
                if unknown_oracle_test(s_trial_1):
                    solution_found = True
                    break
                n_orac_current = random.randint(1, n_iter)
                n_orac += n_orac_current
                circ = grover.circuit(
                    n, k, oracle, n_orac_current, iteration_number_forced=True
                )
                TestGrover.simulator.run(circ)
                s_trial_2 = bin(int(circ[:n]))[2:].rjust(n, "0")
                if unknown_oracle_test(s_trial_2):
                    solution_found = True
                    break
                n_iter = math.ceil(n_iter * c)
            print(f"[{solution_found:5}]oracle used: {n_orac}")
            if not solution_found:
                error += 1
        error_rate = error / M
        print(
            "for n = %d, %d errors in %d tests, error rate = %f"
            % (n, error, M, error_rate)
        )
        if error_rate > 1 - 1 / 8:
            assert 0

    def test_partial_grover(self):
        n_block = 3
        n = 5
        print("run with n = ", n)
        error = 0
        N = 2 ** n
        for target in range(0, N):
            f = [target]
            k, oracle = main_oracle(n, f)
            result = PartialGrover(simulator=TestGrover.simulator).run(
                n, n_block, k, oracle
            )
            if (target >> (n - k)) != (result >> (n - k)):
                error += 1
        error_rate = error / N
        print(
            "for n = %d, %d errors in %d tests, error rate = %f"
            % (n, error, N, error / N)
        )
        if error_rate > 1 / math.sqrt(2 ** n_block):
            assert 0


if __name__ == "__main__":
    unittest.main()
