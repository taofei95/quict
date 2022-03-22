#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

from QuICT.algorithm.quantum_algorithm import (
    BEAShorFactor,
    HRSShorFactor
)


def test_BEAShorFactor():
    from QuICT.simulation.cpu_simulator import CircuitSimulator
    simulator = CircuitSimulator()
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = BEAShorFactor.run(N=number, max_rd=10, simulator=simulator)
        assert number % a == 0


test_BEAShorFactor()
