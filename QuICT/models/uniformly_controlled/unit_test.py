#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/10 8:41 上午
# @Author  : Han Yu
# @File    : unit_test.py

import pytest
import random
import numpy as np
from QuICT.models import *
from QuICT.algorithm import SyntheticalUnitary
from QuICT.models._gate import GATE_ID


def test_multifold():
    types = [GATE_ID["Ry"], GATE_ID["Rz"], GATE_ID["Rx"]]
    max_test = 5
    every_round = 20
    for i in range(1, max_test):
        for type in types:
            for _ in range(every_round):
                circuit = Circuit(i + 1)
                pargs = []
                for _ in range(1 << i):
                    pargs.append(np.pi / 2)
                MultifoldControlledRotation(pargs, type) | circuit
                unitary = SyntheticalUnitary.run(circuit, showSU = False)
                if type == GATE_ID["Rx"]:
                    gate = Rx
                elif type == GATE_ID["Ry"]:
                    gate = Ry
                else:
                    gate = Rz
                for h in range(1 << (i + 1)):
                    for l in range(1 << (i + 1)):
                        if (h >> 1) != (l >> 1):
                            if abs(abs(unitary[h, l])) > 1e-10:
                                assert 0

                dets = []
                for uni in range(1 << i):
                    part = unitary[uni * 2:uni * 2 + 2, uni * 2:uni * 2 + 2]
                    comparision = np.mat(gate(pargs[uni]).matrix.reshape(2, 2))

                    det1 = np.linalg.det(part)
                    n = np.shape(part)[0]
                    det1 = np.power(det1, 1 / n)
                    part[:] /= det1

                    det2 = np.linalg.det(comparision)
                    n = np.shape(comparision)[0]
                    det2 = np.power(det2, 1 / n)
                    comparision[:] /= det2

                    dets.append(det1 / det2)

                    if (abs(abs(part - comparision)) > 1e-10).any():
                        assert 0

                for index in range(len(dets) - 1):
                    if abs(abs(dets[index + 1] - dets[index])) > 1e-10:
                        assert 0
    assert 1

if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
