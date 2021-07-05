#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/3 4:04 下午
# @Author  : Han Yu
# @File    : circuit_draw

from QuICT.core import *

if __name__ == "__main__":
    circuit = Circuit(5)
    circuit.random_append(50)
    circuit.draw()
    circuit.draw(method='command')
