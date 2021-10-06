#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

from QuICT.algorithm import ClassicalShorFactor
from QuICT.algorithm import ClassicalZipShorFactor
from QuICT.algorithm import ShorFactor
from QuICT.algorithm import ZipShorFactor
from QuICT.algorithm import BEAShorFactor
from QuICT.algorithm import HRSShorFactor


def test_ShorFactor():
    a, _, _, _, _ = ShorFactor.run(15)
    assert 15 % a == 0


def test_ZipShorFactor():
    test_list = [15, 57]
    for number in test_list:
        a, _, _, _, _ = ZipShorFactor.run(number)
        assert number % a == 0


def test_ClassicalShorFactor():
    number_list = [
        2, 4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
        30, 32, 33, 34, 35, 36,
        45, 51, 55, 57, 95, 85
    ]
    for number in number_list:
        a, _, _, _, _ = ClassicalShorFactor.run(number)
        assert number % a == 0


def test_ClassicalZipShorFactor():
    number_list = [
        2, 4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
        30, 32, 33, 34, 35, 36,
        45, 51, 55, 57, 95, 85
    ]
    for number in number_list:
        a, _, _, _, _ = ClassicalZipShorFactor.run(number)
        assert number % a == 0


def test_BEAShorFactor():
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
        30, 32, 33, 34, 35, 36,
        45, 51, 55, 57, 95, 85,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = BEAShorFactor.run(number)
        print(a)
        assert number % a == 0


def test_HRSShorFactor():
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
        30, 32, 33, 34, 35, 36,
        45, 51, 55, 57, 95, 85,
    ]
    for number in number_list:
        print('--------------------------------------------')
        a = HRSShorFactor.run(number)
        assert number % a == 0


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])