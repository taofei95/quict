#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

from QuICT.algorithm.quantum_algorithm import (
    ClassicalShorFactor,
    ClassicalZipShorFactor,
    ShorFactor,
    ZipShorFactor,
    BEAShorFactor,
    HRSShorFactor
)


# def test_ShorFactor():
#     a, _, _, _, _ = ShorFactor.run(15)
#     assert 15 % a == 0


# def test_ZipShorFactor():
#     test_list = [15, 57]
#     for number in test_list:
#         a, _, _, _, _ = ZipShorFactor.run(number)
#         assert number % a == 0


# def test_ClassicalShorFactor():
#     number_list = [
#         2, 4, 6, 8, 9, 10,
#         12, 14, 15, 16, 18, 20,
#         21, 22, 24, 25, 26, 27,
#         30, 32, 33, 34, 35, 36,
#         45, 51, 55, 57, 95, 85
#     ]
#     for number in number_list:
#         a, _, _, _, _ = ClassicalShorFactor.run(number)
#         assert number % a == 0


# def test_ClassicalZipShorFactor():
#     number_list = [
#         2, 4, 6, 8, 9, 10,
#         12, 14, 15, 16, 18, 20,
#         21, 22, 24, 25, 26, 27,
#         30, 32, 33, 34, 35, 36,
#         45, 51, 55, 57, 95, 85
#     ]
#     for number in number_list:
#         a, _, _, _, _ = ClassicalZipShorFactor.run(number)
#         assert number % a == 0

#TODO: test when Unitary Simulator satisfies n-qubit(n>2) gate (e.g. CSwap)
# def test_BEAShorFactor():
#     number_list = [
#         4, 6, 8, 9, 10,
#         12, 14, 15, 16, 18, 20,
#         21, 22, 24, 25, 26, 27,
#         30, 32, 33, 34, 35, 36,
#         # 45, 51, 55, 57, 95, 85,
#     ]
#     for number in number_list:
#         print('-------------------FACTORING %d-------------------------' % number)
#         a = BEAShorFactor.run(N=number, max_rd=10)
#         print(a)
#         assert number % a == 0


# def test_HRSShorFactor():
#     number_list = [
#         4, 6, 8, 9, 10,
#         12, 14, 15, 16, 18, 20,
#         21, 22, 24, 25, 26, 27,
#         30, 32, 33, 34, 35, 36,
#         # 45, 51, 55, 57, 95, 85,
#     ]
#     for number in number_list:
#         print('-------------------FACTORING %d-------------------------' % number)
#         a = HRSShorFactor.run(number,10)
#         assert number % a == 0

# def test_BEAShorFactor():
#     number_list = [
#         4, 6, 8, 9, 10,
#         12, 14, 15, 16, 18, 20,
#         21, 22, 24, 25, 26, 27,
#     ]
#     for number in number_list:
#         print('-------------------FACTORING %d-------------------------' % number)
#         a = BEAShorFactor.run(N=number, max_rd=10)
#         assert number % a == 0

def test_BEAShorFactor_on_ConstantStateVectorSimulator():
    from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
    simulator = ConstantStateVectorSimulator(
        precision="double",
        gpu_device_id=0,
        sync=True
    )
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = BEAShorFactor.run(N=number, max_rd=10, simulator=simulator)
        assert number % a == 0

def test_HRSShorFactor_on_ConstantStateVectorSimulator():
    from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
    simulator = ConstantStateVectorSimulator(
        precision="double",
        gpu_device_id=0,
        sync=True
    )
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = HRSShorFactor.run(N=number, max_rd=10, simulator=simulator)
        assert number % a == 0

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])