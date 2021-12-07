#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/9/27 上午10:10
# @Author  : Kaiqi Li
# @File    : statevector_simulator_unit_test.py

import pytest
import os
import numpy as np

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.simulation import ConstantStateVectorSimulator


@pytest.mark.skipif(os.environ.get("test_with_gpu", False), reason="Required GPU support")
def test_constant_statevectorsimulator():
    qubit_num = 10

    circuit = Circuit(qubit_num)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit

    simulator = ConstantStateVectorSimulator(
        precision=np.complex128,
        gpu_device_id=0,
        sync=True)
    state = simulator.run(circuit)

    state_expected = Amplitude.run(circuit)

    assert np.allclose(state.get(), state_expected)


if __name__ == "__main__":
    pytest.main([".statevector_simulator_unit_test.py"])
