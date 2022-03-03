#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/11/12 上午10:10
# @Author  : Kaiqi Li
# @File    : multigpu_simulator_unit_test.py

import os
import unittest
from concurrent.futures import ProcessPoolExecutor, as_completed

from cupy.cuda import nccl

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.utility import Proxy
from QuICT.simulation.gpu_simulator import MultiStateVectorSimulator


q = 5
CIRCUIT = Circuit(q)
QFT.build_gate(q) | CIRCUIT
QFT.build_gate(q) | CIRCUIT
QFT.build_gate(q) | CIRCUIT


def worker(ndev, uid, dev_id):
    proxy = Proxy(ndevs=ndev, uid=uid, dev_id=dev_id)

    simulator = MultiStateVectorSimulator(
        proxy=proxy,
        precision="double",
        gpu_device_id=dev_id,
        sync=True
    )
    state = simulator.run(CIRCUIT)

    return state.get()


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestMultiSVSimulator(unittest.TestCase):
    def test_simulator(self):
        ndev = 2
        uid = nccl.get_unique_id()
        with ProcessPoolExecutor(max_workers=ndev) as executor:
            tasks = [
                executor.submit(worker, ndev, uid, dev_id) for dev_id in range(ndev)
            ]

        results = []
        for t in as_completed(tasks):
            results.append(t.result())

        assert 1


if __name__ == "__main__":
    unittest.main()
