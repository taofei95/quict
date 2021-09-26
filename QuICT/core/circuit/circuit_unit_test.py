#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/16 1:20 下午
# @Author  : Han Yu
# @File    : circuit_unit_test.py

import random

import pytest

from QuICT import *
from QuICT.algorithm import Amplitude
from .circuit import Circuit

def getRandomList(count, upper_bound):
    _rand = [i for i in range(upper_bound)]
    for i in range(upper_bound - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:count]


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
    n = 10
    circuit = Circuit(n)
    circuit[0:2].force_assign_random()
    circuit[2:3].force_assign_random()
    circuit[3:7].force_assign_random()
    circuit[7:10].force_assign_random()
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


def test_sub_circuit():
    circuit = Circuit(5)
    CX | circuit([0, 1])
    CX(name="AA") | circuit([1, 2])
    CX | circuit([2, 3])
    circuit.sub_circuit(slice(4), start="AA", max_size=1, remove=True).print_information()
    circuit.print_information()
    new_circuit = Circuit(6)
    circuit | new_circuit
    new_circuit.print_information()
    assert 1


def test_sub_circuit_local():
    circuit = Circuit(5)
    CX | circuit([0, 1])
    CX | circuit([2, 1])
    CX | circuit([1, 0])
    circuit.sub_circuit(slice(2), local=True, remove=True).print_information()
    circuit.print_information()
    assert 1


if __name__ == "__main__":
    pytest.main(["./circuit_unit_test.py"])
