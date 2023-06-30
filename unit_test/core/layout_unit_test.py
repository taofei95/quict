#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 8:32 下午
# @Author  : Han Yu
# @File    : topology_unit_test.py

import os
import random

import pytest

from QuICT.core import Layout


def get_random_list(count, upper_bound):
    _rand = [i for i in range(upper_bound)]
    for i in range(upper_bound - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:count]


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
            out_list = get_random_list(2, i)
            layout.add_edge(out_list[0], out_list[1], random.random())
            assert layout.check_edge(out_list[0], out_list[1])


def test_load():
    layout = Layout.load_file(
        os.path.dirname(os.path.abspath(__file__))
        + "/../../example/layout/ibmqx2_layout.json"
    )
    assert layout.name == "ibmqx2"
    assert layout.qubit_number == 5
    assert len(layout.edge_list) == 6
    for edge in layout:
        assert not edge.directional


def test_store():
    for i in range(2, 10):
        layout = Layout(i)
        for _ in range(200):
            out_list = get_random_list(2, i)
            layout.add_edge(out_list[0], out_list[1], random.random())
        another_layout = Layout.from_json(layout.to_json())
        for edge in layout:
            assert another_layout.check_edge(edge.u, edge.v)


def test_special_build():
    # linear layout build 0 - 1 - 2 - 3 - 4
    lin_layout = Layout.linear_layout(qubit_number=5)
    assert lin_layout.check_edge(0, 1) and not lin_layout.check_edge(2, 4)

    # grid layout
    #   0 - 1 - 2
    #   |   |   |
    #   3 - 4 - 5
    #   |   |   |
    #   6 - 7 - 8
    gird_layout = Layout.grid_layout(9, width=3)
    assert gird_layout.check_edge(4, 7) and not gird_layout.check_edge(1, 5)

    # rhombus layout
    #    0     1     2
    #  /   \ /   \  /
    # 3     4     5
    #  \   / \   /  \
    #    6     7     8
    rhombus_layout = Layout.rhombus_layout(9, width=3)
    assert rhombus_layout.check_edge(0, 4) and not rhombus_layout.check_edge(7, 8)


if __name__ == "__main__":
    pytest.main()
