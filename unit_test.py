#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 10:10 上午
# @Author  : Dang Haoran
# @File    : unit_test.py


from time import time
import numpy as np

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.qcda.simulation.statevector_simulator.constant_statevecto_simulator import ConstantStateVectorSimulator


"""
the file describe Simulators between two basic gates.
"""


def test_constant_statevectorsimulator():
    # pre-compiled kernel function
    qubit = 29
    circuit = Circuit(qubit)
    X | circuit
    Measure | circuit
    now = ConstantStateVectorSimulator(circuit, np.complex64)
    now.run()

test_constant_statevectorsimulator()
