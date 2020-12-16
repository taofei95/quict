#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/14 10:34 上午
# @Author  : Han Yu
# @File    : Efficiency_Checker.py

from QuICT.tools.checker import StandardEfficiencyChecker

StandardEfficiencyChecker.min_qubits = 10
StandardEfficiencyChecker.max_qubits = 20
StandardEfficiencyChecker.size = 1000
StandardEfficiencyChecker.run()
