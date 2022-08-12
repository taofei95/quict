#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/13 1:35 下午
# @Author  : Han Yu
# @File    : entir_test.py

import os
import pytest
import random

import numpy as np

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.algorithm import *
from QuICT.qcda.optimization.template_optimization.templates import *
from QuICT.qcda.optimization.template_optimization import TemplateOptimization


def mat_from_circuit(circuit):
    n = circuit.circuit_width()
    mat = np.identity(n, dtype=np.bool)
    for gate in circuit.gates:
        if gate.qasm_name == 'x':
            target = gate.targ
            for i in range(n):
                mat[target, i] = not mat[target, i]
        elif gate.qasm_name == 'cx':
            control = gate.carg
            target = gate.targ
            for i in range(n):
                if mat[control, i]:
                    mat[target, i] = not mat[target, i]
        elif gate.qasm_name == 'ccx':
            control1 = gate.cargs[0]
            control2 = gate.cargs[1]
            target = gate.targ
            for i in range(n):
                if mat[control1, i] and mat[control2, i]:
                    mat[target, i] = not mat[target, i]
        else:
            raise Exception("wuhu")
    return mat


def equiv(circuit1, circuit2):
    if circuit1.width() != circuit2.width():
        return False

    return np.allclose(circuit1.matrix(), circuit2.matrix())


def test_can_run():
    names = []
    for root, dirs, files in os.walk('./templates/nct'):
        for name in files:
            if name == '__init__.py' or not name.endswith('.py'):
                continue
            name = name[:-3]
            names.append(name)

    for i in range(3, 4):
        circuit = Circuit(i)
        circuit.random_append(100, typelist=[GateType.x, GateType.cx, GateType.ccx])

        templates = []
        for name in names:
            templates.append(eval(name)())
        circuit_opt = TemplateOptimization.execute(circuit, templates)
        equ = equiv(circuit, circuit_opt)
        if not equ:
            circuit.print_information()
            circuit_opt.print_information()
            print(len(circuit.gates), len(circuit_opt.gates))
            assert 0

        circuit_opt_opt = TemplateOptimization.execute(circuit_opt, templates)
        equ = equiv(circuit_opt_opt, circuit_opt)
        if not equ:
            circuit.print_information()
            circuit_opt.print_information()
            print(len(circuit.gates), len(circuit_opt.gates))
            assert 0


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
