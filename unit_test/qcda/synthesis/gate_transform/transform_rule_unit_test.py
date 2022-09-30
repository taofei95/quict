#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/28 1:25 上午
# @Author  : Han Yu
# @File    : unit_test

import numpy as np
from scipy.stats import unitary_group

from QuICT.core.gate import build_gate
from QuICT.qcda.synthesis.gate_transform.transform_rule import *


def test_one_qubit_rules():
    for _ in range(20):
        mat = unitary_group.rvs(2)
        for rule in [xyx_rule, zyz_rule, ibmq_rule]:
            unitary = Unitary(mat) & 0
            gates = rule(unitary)
            phase = np.dot(gates.matrix(), np.linalg.inv(mat))
            assert np.allclose(phase, phase * np.identity(2))


def test_two_qubit_rules():
    typelist = [
        GateType.cx,
        GateType.cy,
        GateType.cz,
        GateType.ch,
        GateType.crz,
        GateType.rxx,
        GateType.ryy,
        GateType.rzz,
        GateType.fsim
    ]
    for source in typelist:
        for target in typelist:
            if source != target:
                rule = eval(f"{source.name}2{target.name}_rule")
                gate = build_gate(source, [0, 1])
                if gate.params:
                    gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))
                gates = rule(gate)
                assert np.allclose(gate.matrix, gates.matrix())
