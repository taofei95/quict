#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/13 1:35 下午
# @Author  : Han Yu
# @File    : entir_test.py

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import GateType
from QuICT.qcda.optimization.template_optimization.templates import *
from QuICT.qcda.optimization.template_optimization import TemplateOptimization


def test():
    templates = [template_nct_2a_1(), template_nct_2a_2(), template_nct_2a_3()]

    for i in range(3, 6):
        for _ in range(10):
            circuit = Circuit(i)
            circuit.random_append(100, typelist=[GateType.x, GateType.cx, GateType.ccx])

            circuit_opt = TemplateOptimization.execute(circuit)
            mat = circuit.matrix()
            mat_opt = circuit_opt.matrix()
            assert np.allclose(mat, mat_opt)
