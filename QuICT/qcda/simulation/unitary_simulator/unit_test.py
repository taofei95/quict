#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py

import pytest
from QuICT.qcda.simulation.unitary_simulator import *
from QuICT.core import *

"""
the file describe Simulators between two basic gates.
"""

def test_merge_two_unitary():
    targs = [0, 1]
    compositeGate1 = CZ & targs
    compositeGate2 = X & targs[1]
    print(UnitarySimulator.merge_two_unitary(compositeGate1, compositeGate2).compute_matrix)
    print("\nfinish\n")


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
    assert 0
    #test_merge_two_unitary()
