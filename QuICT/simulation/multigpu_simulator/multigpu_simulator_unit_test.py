#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/9/27 上午10:10
# @Author  : Kaiqi Li
# @File    : statevector_simulator_unit_test.py

import pytest
import os
import numpy as np
import multiprocessing
from cupy.cuda import nccl

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.ops.utils import Proxy
from QuICT.simulation import MultiStateVectorSimulator


def worker(uid, ndevs, dev_id, circuit):
    proxy = Proxy(ndevs=ndevs, uid=uid, rank=dev_id)

    simulator = MultiStateVectorSimulator(
        proxy=proxy,
        circuit=circuit,
        precision=np.complex128,
        gpu_device_id=dev_id,
        sync=True
    )
    state = simulator.run()

    state_expected = Amplitude.run(circuit)

    half_point = len(state_expected) << 1

    if dev_id:
        assert np.allclose(state.get(), state_expected[half_point:])
    else:
        assert np.allclose(state.get(), state_expected[:half_point])


def multigpu_simulator_start(qubit_num):
    multiprocessing.set_start_method("spawn")

    uid = nccl.get_unique_id()

    circuit = Circuit(qubit_num)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit

    p1 = multiprocessing.Process(target=worker, args=(uid, 2, 0, circuit,))
    p2 = multiprocessing.Process(target=worker, args=(uid, 2, 1, circuit,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


@pytest.mark.skipif(os.environ.get("test_with_gpu", False), reason="Required GPU support")
def test_multigpu_statevectorsimulator():
    qubit_num = 10

    multigpu_simulator_start(qubit_num)


if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    pytest.main([".multigpu_simulator_unit_test.py"])
