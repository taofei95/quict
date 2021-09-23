#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 8:32 下午
# @Author  : Han Yu
# @File    : topology_unit_test.py

import os
import pytest
import random

from QuICT.core import Layout, LayoutEdge

def getRandomList(l, n):
    _rand = [i for i in range(n)]
    for i in range(n - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:l]

def test_build():
    layout = Layout(10)
    layout.add_edge(1, 5, 0.8)
    layout.add_edge(1, 4, 0.9)
    layout.add_edge(2, 4, 0.9)
    layout.add_edge(4, 5, 0.9)
    out_list = layout.out_edges(1)
    assert len(out_list) == 2
    assert layout.check_edge(1, 5)
    assert not layout.check_edge(1, 3)
    # layout.write_file()

def test_random_build():
    for i in range(2, 10):
        layout = Layout(i)
        for _ in range(200):
            out_list = getRandomList(2, i)
            layout.add_edge(out_list[0], out_list[1], random.random())
            assert layout.check_edge(out_list[0], out_list[1])

def test_load():
    layout = Layout.load_file(os.path.dirname(os.path.abspath(__file__)) + "/../../../example/layout/ibmqx2.layout")
    assert layout.name == 'ibmqx2'
    assert layout.qubit_number == 5
    assert len(layout.edge_list) == 6

if __name__ == "__main__":
    pytest.main(["./topology_unit_test.py"])
