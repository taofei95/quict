#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 1:20 下午
# @Author  : Li Kaiqi
# @File    : circuit_unit_test.py

import unittest
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.noise import *
from QuICT.core.operator import NoiseGate
from QuICT.simulation.density_matrix import DensityMatrixSimulator


class TestNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Noise unit test start!")
        cls.circuit = Circuit(4)
        H | cls.circuit(0)
        CX | cls.circuit([0, 1])
        CX | cls.circuit([1, 2])
        CX | cls.circuit([2, 3])

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Noise unit test finished!")

    def test_pauilerror(self):
        pauil_error_rate = 0.4
        # bitflip pauilerror
        bf_err = BitflipError(pauil_error_rate)

        # phaseflip pauilerror
        pf_err = PhaseflipError(pauil_error_rate)

        # 2-bits pauilerror
        bits_err = PauliError(
            [('xy', pauil_error_rate), ('zi', 1 - pauil_error_rate)],
            num_qubits=2
        )

        # tensor pauil error
        tensor_error = bf_err.tensor(bf_err)

        # build noise model
        nm = NoiseModel()
        nm.add(bits_err, ['cx'], [0, 1])

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulator(accumulated_mode=True)
        _ = dm_simu.run(TestNoise.circuit, quantum_machine_model=nm)
        count = dm_simu.sample(100)

        assert count[0] + count[15] + count[7] + count[8] == 100

    def test_depolarizingerror(self):
        depolarizing_rate = 0.05
        # 1-qubit depolarizing error
        single_dep = DepolarizingError(depolarizing_rate, num_qubits=1)

        # 2-qubits depolarizing error
        double_dep = DepolarizingError(depolarizing_rate, num_qubits=2)

        # tensor depolarizing error
        tensor_error = single_dep.tensor(single_dep)

        # build noise model
        nm = NoiseModel()
        nm.add(tensor_error, ['cx'], [0, 1])

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulator()
        _ = dm_simu.run(TestNoise.circuit, quantum_machine_model=nm)
        count = dm_simu.sample(100)

        assert count[0] + count[15] + count[7] + count[8] == 100

    def test_damping(self):
        # Amplitude damping error
        amp_err = DampingError(amplitude_prob=0.2, phase_prob=0, dissipation_state=0.3)
        # Phase damping error
        phase_err = DampingError(amplitude_prob=0, phase_prob=0.5)
        # Amp + Phase damping error
        amp_phase_err = DampingError(amplitude_prob=0.1, phase_prob=0.3, dissipation_state=0.5)

        # tensor damping error
        tensor_error = amp_err.tensor(phase_err)

        # build noise model
        nm = NoiseModel()
        nm.add(tensor_error, ['cx'], [0, 1])

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulator()
        _ = dm_simu.run(TestNoise.circuit, quantum_machine_model=nm)
        count = dm_simu.sample(100)

        assert count[0] + count[15] + count[7] + count[8] == 100

    def test_readout(self):
        # single-qubit Readout Error
        single_readout = ReadoutError(np.array([[0.8, 0.2], [0.2, 0.8]]))
        # double-qubits Readout Error
        double_readout = ReadoutError(
            np.array(
                [[0.7, 0.1, 0.1, 0.1],
                 [0.1, 0.7, 0.1, 0.1],
                 [0.1, 0.1, 0.7, 0.1],
                 [0.1, 0.1, 0.1, 0.7]]
            )
        )

        comp_readout = single_readout.compose(double_readout)
        tensor_readout = single_readout.tensor(single_readout)

        # build noise model
        nm = NoiseModel()
        nm.add_readout_error(single_readout, [1, 3])
        nm.add_readout_error(double_readout, [1, 3])

        dm_simu = DensityMatrixSimulator(accumulated_mode=True)
        _ = dm_simu.run(self.circuit, quantum_machine_model=nm)
        count = dm_simu.sample(100)

        assert count[0] + count[1] + count[4] + count[5] + \
            count[10] + count[11] + count[14] + count[15] == 100

    def test_noisemodel(self):
        pauil_error_rate = 0.4
        bf_err = BitflipError(pauil_error_rate)
        pf_err = PhaseflipError(pauil_error_rate)
        single_readout = ReadoutError(np.array([[0.8, 0.2], [0.2, 0.8]]))

        # 2-bits pauilerror
        bits_err = PauliError(
            [('zy', pauil_error_rate), ('xi', 1 - pauil_error_rate)],
            num_qubits=2
        )

        # build noise model
        nm = NoiseModel()
        nm.add_noise_for_all_qubits(bf_err, ['h'])
        nm.add(pf_err, ['z'], [0, 1, 3])
        nm.add(bits_err, ['cx'], [0, 1])
        nm.add_readout_error(single_readout, [1, 2])

        # build test circuit
        cir = Circuit(4)
        H | cir
        Z | cir(2)
        Z | cir(3)
        CX | cir([2, 1])
        CX | cir([0, 1])
        cir_size = cir.size()

        noised_circuit = nm.transpile(cir, accumulated_mode=True)
        sum_ng = 0
        for ng in noised_circuit.gates:
            if isinstance(ng, NoiseGate):
                sum_ng += 1

        assert sum_ng == 6      # 4 for H; 1 for Z; 1 for CX

        noised_circuit = nm.transpile(cir)
        assert noised_circuit.size() == cir_size + sum_ng


if __name__ == "__main__":
    unittest.main()
