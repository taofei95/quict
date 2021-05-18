#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 9:58
# @Author  : Han Yu
# @File    : unit_test.py

import pytest

from QuICT.algorithm import Amplitude
from QuICT.core import *

def test_amplitude():
    for i in range(1, 10):
        circuit = Circuit(i)
        for j in range(i):
            H | circuit(j)
        amplitude = Amplitude.run(circuit)
        if len(amplitude) != (1 << i):
            assert 0
        for k in range(1 << i):
            if abs(abs(amplitude[k]) * abs(amplitude[k]) - (1.0 / (1 << i))) > 1e-6:
                print(i, k, abs(amplitude[k]) * abs(amplitude[k]), (1.0 / (1 << i)))
                assert 0
        assert 1

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
