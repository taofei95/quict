#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/11/12 上午10:10
# @Author  : Kaiqi Li
# @File    : multigpu_simulator_unit_test.py

import os
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed

from QuICT.core import Circuit
from QuICT.core.gate import *


if os.environ.get("test_with_gpu"):
    from cupy.cuda import nccl
    from QuICT.utility import Proxy
    from QuICT.simulation.state_vector.gpu_simulator import MultiDeviceSimulatorLauncher


q = 5
CIRCUIT = Circuit(q)
QFT.build_gate(q) | CIRCUIT
QFT.build_gate(q) | CIRCUIT
QFT.build_gate(q) | CIRCUIT


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestMultiSVSimulator(unittest.TestCase):
    def test_simulator(self):
        ndev = 2
        md_sim = MultiDeviceSimulatorLauncher(ndev=ndev)
        _ = md_sim.run(circuit=CIRCUIT)

        assert 1


if __name__ == "__main__":
    unittest.main()
