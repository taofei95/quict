#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/23 4:31 下午
# @Author  : Han Yu
# @File    : qubit_unit_test.py

import pytest

from .._qubit import Qubit, Qureg, Tangle
from QuICT.algorithm import Amplitude
from QuICT.models import Circuit, X, H, Measure

def test_Qubit_Attributes_prob():
    circuit = Circuit(3)
    X       | circuit(0)
    Measure | circuit(0)
    circuit.flush()
    if circuit[0].measured != 1:
        assert 0
    if abs(circuit[0].prob - 1) > 1e-10:
        assert 0
    H       | circuit(1)
    Measure | circuit(1)
    circuit.flush()
    if abs(circuit[1].prob - 0.5) > 1e-10:
        print(circuit[1].prob)
        assert 0
