#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/12 6:17 下午
# @Author  : Han Yu
# @File    : hy_unit_test.py

import os
import pytest
from templates import *
from template_optimization import TemplateOptimization
from template_matching import TemplateMatching, TemplateSubstitution, MaximalMatches
from template_matching.dagdependency import DAGDependency, circuit_to_dagdependency


def test_can_run():
    names = []
    for root, dirs, files in os.walk('./templates/nct'):
        for name in files:
            if name == '__init__.py' or not name.endswith('.py'):
                continue
            name = name[:-3]
            names.append(name)
    for root, dirs, files in os.walk('./templates/cnot_templates'):
        for name in files:
            if name == '__init__.py' or not name.endswith('.py'):
                continue
            name = name[:-3]
            names.append(name)
    for name1 in names:
        for name2 in names:
            circuitC = eval(name1)()
            circuitT = eval(name2)()
            if circuitT.circuit_length() > circuitC.circuit_length():
                continue
            dag_C = circuit_to_dagdependency(circuitC)
            dag_T = circuit_to_dagdependency(circuitT)
            template_m = TemplateMatching(dag_C, dag_T)
            circuitC.print_infomation()
            circuitT.print_infomation()
            template_m.run_template_matching()
            matches = template_m.match_list
            if matches:
                maximal = MaximalMatches(matches)
                maximal.run_maximal_matches()
                max_matches = maximal.max_match_list
                for match in max_matches:
                    print(match.match, match.qubit, '\n')
    assert 1

if __name__ == '__main__':
    pytest.main(["./hy_unit_test.py"])
