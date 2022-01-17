#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/9/27 上午10:10
# @Author  : Kaiqi Li
# @File    : statevector_simulator_unit_test.py

import pytest
import os

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


@pytest.mark.skipif(os.environ.get("test_with_gpu", False), reason="Required GPU support")
def test_constant_statevectorsimulator():
    qubit_num = 5

    circuit = Circuit(qubit_num)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit

    simulator = ConstantStateVectorSimulator(
        precision="double",
        gpu_device_id=0,
        sync=True
    )
    state = simulator.run(circuit)

    assert 1


if __name__ == "__main__":
    pytest.main([".statevector_simulator_unit_test.py"])
