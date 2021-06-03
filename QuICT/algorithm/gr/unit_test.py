#!/usr/bin/env python
# -*- coding:utf8 -*-

import pytest

from .gr import *
from QuICT.core import *

def main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)

def test_1():#std grover
    for test_number in range(3, 5):
        for i in range(8):
            test = [0] * (1 << test_number)
            test[i] = 1
            ans = StandardGrover.run(test, main_oracle)
            # print("[%2d in %2d]answer:%2d"%(i,1<<test_number,ans))

if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
