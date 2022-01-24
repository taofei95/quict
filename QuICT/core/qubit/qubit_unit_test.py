#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/23 4:31
# @Author  : Li Kaiqi
# @File    : qubit_unit_test.py
import pytest
import random

from QuICT.core import Qureg


def test_qubit_attr():
    # ID unique
    qureg = Qureg(10)
    id_list = [qubit.id for qubit in qureg]

    assert len(set(id_list)) == len(id_list)

    # measure
    for qubit in qureg:
        measure = random.random() > 0.5
        if measure:
            qubit.measured = 1
        else:
            qubit.measured = 0

        assert int(qubit) == qubit.measured
        assert bool(qubit) == measure


def test_qureg_call():
    qureg = Qureg(10)
    idx = random.sample(range(10), 3)
    squbit_ids = [qureg[i].id for i in idx]

    cqureg = qureg(idx)
    for cq in cqureg:
        assert cq.id in squbit_ids


def test_qureg_int():
    qureg = Qureg(3)
    for qubit in qureg:
        qubit.measured = 1

    assert int(qureg) == 7


def test_qureg_operation():
    # add
    q1 = Qureg(5)
    q2 = Qureg(5)
    q_add = q1 + q2

    assert len(q_add) == (len(q1) + len(q2))

    # equal
    assert q1 == q1
    assert not q1 == q2

    # diff
    diff_q = q1.diff(q2)
    assert diff_q == q1


if __name__ == "__main__":
    pytest.main(["./qubit_unit_test.py"])
