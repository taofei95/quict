#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/3 4:04 下午
# @Author  : Han Yu
# @File    : circuit_draw

from QuICT.core import Circuit

circuit = Circuit(5)
circuit.random_append(50)
circuit.draw(filename="random")
