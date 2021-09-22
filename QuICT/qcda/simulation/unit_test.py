#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 1:06 下午
# @Author  : Han Yu
# @File    : unit_test

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *

from QuICT.ops.linalg import *

from ._simulation import BasicSimulator


def test_pretreatment():
    circuit = Circuit(10)
    circuit.random_append(100, typeList=[GATE_ID["CX"], GATE_ID["X"]])
    pretreatment = BasicSimulator.pretreatment(circuit)
    unitary1 = SyntheticalUnitary.run(circuit)
    pretreatment.print_information()
    unitary2 = pretreatment.matrix()
    assert np.allclose(unitary1, unitary2)

def test_permutation():
    array = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    new_array = MatrixPermutation(array, np.array([1, 0]))
    print(new_array)
    assert not np.allclose(new_array, array)
