#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/16 1:20 下午
# @Author  : Han Yu
# @File    : circuit_unit_test.py

import pytest
import random

import numpy as np

from QuICT import *
from QuICT.algorithm import Amplitude

def getRandomList(l, n):
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:l]

def test_pratial_prob_whole():
    for n in range(1, 10):
        circuit = Circuit(n)
        circuit.assign_initial_random()

        order = getRandomList(n, n)
        reverse_order = [i for i in range(n)]
        for i in range(len(order)):
            reverse_order[order[i]] = i
        # order = [i for i in range(n)]
        prob = circuit.partial_prob(order)
        amplitude = Amplitude.run(circuit)
        for position in range(1 << n):
            prob_position = 0
            for bit in range(n):
                if (1 << (n - bit - 1)) & position != 0:
                    prob_position += 1 << (n - reverse_order[bit] - 1)
            if abs(abs(prob[prob_position]) - abs(amplitude[position]) * abs(amplitude[position])) > 1e-6:
                assert 0

def test_pratial_prob_part():
    circuit = Circuit(10)
    circuit[0:2].assign_initial_random()
    circuit[2:3].assign_initial_random()
    circuit[3:7].assign_initial_random()
    circuit[7:10].assign_initial_random()
    order = [i for i in range(n)]
    reverse_order = [i for i in range(n)]
    for i in range(len(order)):
        reverse_order[order[i]] = i
    prob = circuit.partial_prob(order)
    amplitude = Amplitude.run(circuit)
    for position in range(1 << n):
        prob_position = 0
        for bit in range(n):
            if (1 << (n - bit - 1)) & position != 0:
                prob_position += 1 << (n - reverse_order[bit] - 1)
        if abs(abs(prob[prob_position]) - abs(amplitude[position]) * abs(amplitude[position])) > 1e-6:
            assert 0

if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    # pytest.main(["./circuit_unit_test.py"])
    test_pratial_prob()

