#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/1/3 2:22 下午
# @Author  : Han Yu
# @File    : test.py

from QuICT import *
from QuICT.algorithm import SyntheticalUnitary
from QuICT.qcda.synthesis import uniformlyUnitary

circuit = Circuit(4)
unitary1 = U3([0, 0, 0]).matrix
unitary2 = U3([0, 0, 0]).matrix
unitary3 = U3([0, 0, 0]).matrix
unitary4 = U3([0, 0, 0]).matrix
unitary5 = U3([0, 0, 0]).matrix
unitary6 = U3([0, 0, 0]).matrix
unitary7 = U3([0, 0, 0]).matrix
unitary8 = U3([0, 0, 0]).matrix
unitaries = [unitary1, unitary2, unitary3, unitary4, unitary5, unitary6, unitary7, unitary8]
uniformlyUnitary(unitaries) | circuit
unitary = SyntheticalUnitary.run(circuit)
print(np.round(unitary, 2))
if abs(unitary[0, 0]) > 1e-10:
    delta = unitary1[0] / unitary[0, 0]
else:
    delta = unitary1[1] / unitary[0, 1]
for j in range(1 << (4 - 1)):
    unitary_slice = unitary[2 * j:2 * (j + 1), 2 * j:2 * (j + 1)]
    unitary_slice[:] *= delta
    phase = np.any(abs(unitary_slice - unitaries[j].reshape(2, 2)) > 1e-10)
    if phase:
        print(unitary_slice)
        assert 0
