#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/28 6:28 下午
# @Author  : Han Yu
# @File    : refine_unit_test

import pytest

import numpy as np

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.simulation.statevector_simulator.refine_statevector_simulator import *
from time import time

from .statevector_simulator import StateVectorSimulatorRefine
from QuICT.ops.linalg.gpu_calculator import *

def test_refine_vec_sim():
    qubit_num = 10
    circuit = Circuit(qubit_num)
    circuit.random_append(5)

    start_time = time()
    state_expected = Amplitude.run(circuit)
    end_time = time()
    duration_2 = end_time - start_time

    start_time = time()
    state = RefineStateVectorSimulator.run(circuit)
    end_time = time()
    duration_1 = end_time - start_time

    assert np.allclose(state, state_expected)
    print()
    print(f"Cur algo: {duration_1} s.")
    print(f"Old algo: {duration_2} s.")
