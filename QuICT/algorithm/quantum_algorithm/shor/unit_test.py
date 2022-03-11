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

#def test_BEAShorFactor_on_UnitarySimulator():
#    number_list = [
#        4, 6, 8, 9, 10,
#        12, 14, 15, 16, 18, 20,
#        21, 22, 24, 25, 26, 27,
#    ]
#    for number in number_list:
#        print('-------------------FACTORING %d-------------------------' % number)
#        a = BEAShorFactor.run(N=number, max_rd=10)
#        assert number % a == 0
#
#def test_HRSShorFactor_on_UnitarySimulator():
#    number_list = [
#        4, 6, 8, 9, 10,
#        12, 14, 15, 16, 18, 20,
#        21, 22, 24, 25, 26, 27,
#    ]
#    for number in number_list:
#        print('-------------------FACTORING %d-------------------------' % number)
#        a = HRSShorFactor.run(N=number, max_rd=10)
#        assert number % a == 0


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