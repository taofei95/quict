#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 1:06 下午
# @Author  : Han Yu
# @File    : unit_test

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *

from ._simulation import BasicGPUSimulator


def test_pretreatment():
    circuit = Circuit(10)
    circuit.random_append(100, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    pretreatment = BasicGPUSimulator.pretreatment(circuit)
    unitary1 = SyntheticalUnitary.run(circuit)
    pretreatment.print_information()
    unitary2 = pretreatment.matrix()
    assert np.allclose(unitary1, unitary2)
