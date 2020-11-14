#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/14 10:34 上午
# @Author  : Han Yu
# @File    : Efficiency_Checker.py

from QuICT.checker import StandardEfficiencyChecker

StandardEfficiencyChecker.min_qubits = 20
StandardEfficiencyChecker.max_qubits = 30
StandardEfficiencyChecker.size = 100
StandardEfficiencyChecker.run()
