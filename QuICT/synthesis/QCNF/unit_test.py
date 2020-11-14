#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 8:36 上午
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random
from QuICT.models import *
from QuICT.synthesis import QCNF

def getRandomList(l, n):
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:l]

def truth(cnf, assignment):
    for req in cnf:
        if req < 0:
            if assignment[abs(req) - 1] == 0:
                return True
        elif req > 0:
            if assignment[req - 1] == 1:
                return True
        else:
            raise Exception("cnf error")
    return False

def test_single_QCNF():
    i = 10
    anc = 4
    circuit = Circuit(i + anc)
    assignment = [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
    i = 0
    for ass in assignment:
        if ass == 1:
            X | circuit(i)
        i += 1

    cnf_list = [[8, 6, -10], [5], [6, -9, -7], [-2, -5, -4], [1, -6], [-1, -6], [6, 8], [2, 3], [7, -1, 9]]
    count = len(cnf_list)

    QCNF(i, count, anc, cnf_list) | circuit
    Measure | circuit(circuit.circuit_length() - 1)
    circuit.flush()
    result = int(circuit(circuit.circuit_length() - 1))
    print(circuit.index_for_qubit(circuit(circuit.circuit_length() - 1)[0]), result)
    flag = 1

    for cnf in cnf_list:
        if not truth(cnf, assignment):
            flag = 0
            break
    if flag != result:
        # print(flag, result, anc, cnf_list, assignment)
        # print(i, count, anc, cnf_list)
        # circuit.print_infomation()
        print(flag, result)
        assert 0
    # circuit.print_infomation()
    # print(i, count, anc, cnf_list)
    assert 0

def w_test_QCNF_small():
    max_cnf_count = 10
    max_qubit = 16
    for i in range(10, 11):
        for k in range(3, i + 1):
            for count in range(max_cnf_count - 1, max_cnf_count):
                cnf_list = []
                for _ in range(count):
                    var_list = getRandomList(random.randint(1, k), i)
                    new_list = []
                    for ss in var_list:
                        new_list.append((ss + 1) * (random.randint(0, 1) * 2 - 1))
                    cnf_list.append(new_list)
                for anc in range(4, max_qubit - i):
                    for _ in range(20):
                        fz = random.randrange(0, 1 << i)
                        circuit = Circuit(i + anc)
                        assignment = []
                        for j in range(i):
                            if ((1 << j) & fz) != 0:
                                X | circuit(j)
                                assignment.append(1)
                            else:
                                assignment.append(0)
                        # circuit.print_infomation()
                        QCNF(i, count, anc, cnf_list) | circuit
                        Measure | circuit(circuit.circuit_length() - 1)
                        circuit.flush()
                        result = int(circuit(circuit.circuit_length() - 1))
                        flag = 1
                        for cnf in cnf_list:
                            if not truth(cnf, assignment):
                                flag = 0
                                break
                        if flag != result:
                            print(flag, result, anc, cnf_list, assignment)
                            print(i, count, anc, cnf_list)
                            circuit.print_infomation()
                            assert 0
    assert 1


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
