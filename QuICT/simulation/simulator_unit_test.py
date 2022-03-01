#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 1:06 下午
# @Author  : Li Kaiqi
# @File    : simulator_unit_test
import os
import unittest

from QuICT.core import Circuit
from QuICT.simulation import Simulator


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGPUSimulator(unittest.TestCase):
    def test_unitary(self):
        circuit = Circuit(10)
        circuit.random_append(100)
        u_sim = Simulator(
            device="GPU",
            backend="unitary"
        )
        _ = u_sim.run(circuit, circuit_out=False, statevector_out=False)

        assert 1

    def test_gpu_statevector(self):
        circuit = Circuit(10)
        circuit.random_append(100)
        
        sv_sim = Simulator(
            device="GPU",
            backend="statevector"
        )
        
        _ = sv_sim.run(circuit, circuit_out=False, statevector_out=False)

        assert 1
        
    def test_multigpu_simulator(self):
        circuit = Circuit(10)
        circuit.random_append(100)
        
        multi_sim = Simulator(
            device="GPU",
            backend="multiGPU",
            ndev=2
        )
        _ = multi_sim.run(circuit, circuit_out=False, statevector_out=False)

        assert 1


class TestCPUSimulator(unittest.TestCase):
    def test_unitary(self):
        circuit = Circuit(10)
        circuit.random_append(100)
        u_sim = Simulator(
            device="CPU",
            backend="unitary"
        )
        _ = u_sim.run(circuit, circuit_out=False, statevector_out=False)

        assert 1
