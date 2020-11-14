#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 10:43 上午
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

def test_1():
    assert 1

def test_2():
    assert 1

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])