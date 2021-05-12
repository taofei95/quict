#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 1:06 下午
# @Author  : Han Yu
# @File    : unit_test

import pytest

import cProfile
import re

from ._simulation import BasicSimulator
from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *

def test_pretreatment():
    circuit = Circuit(30)
    circuit.random_append(1000, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    pretreatment = BasicSimulator.vector_pretreatment(circuit)
    # unitary1 = SyntheticalUnitary.run(circuit)
    # pretreatment.print_information()
    #  unitary2 = pretreatment.matrix()
    # assert np.allcloBasicSimulatorse(unitary1, unitary2)

