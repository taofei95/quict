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
from QuICT.simulation.density_matrix import DensityMatrixSimulation


class TestNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Noise unit test start!")
        cls.circuit = Circuit(3)
        H | cls.circuit
        CX | cls.circuit([1, 2])
        CX | cls.circuit([0, 1])
        CH | cls.circuit([1, 0])
        Swap | cls.circuit([2, 1])
        SX | cls.circuit(0)
        T | cls.circuit(1)
        T_dagger | cls.circuit(0)
        X | cls.circuit(1)
        Y | cls.circuit(1)
        S | cls.circuit(2)
        U1(np.pi / 2) | cls.circuit(2)
        U3(np.pi, 0, 1) | cls.circuit(0)
        Rx(np.pi) | cls.circuit(1)
        Ry(np.pi / 2) | cls.circuit(2)
        Rz(np.pi / 4) | cls.circuit(0)

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

        # build noise model
        nm = NoiseModel()
        nm.add_noise_for_all_qubits(bf_err, ['h'])
        nm.add_noise_for_all_qubits(pf_err, ['x', 'y'])
        nm.add(bits_err, ['cx', 'cz'], [1, 2])

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulation()
        _ = dm_simu.run(TestNoise.circuit, noise_model=nm)

        assert 1

    def test_depolarizingerror(self):
        depolarizing_rate = 0.05
        # 1-qubit depolarizing error
        single_dep = DepolarizingError(depolarizing_rate, num_qubits=1)

        # 2-qubits depolarizing error
        double_dep = DepolarizingError(depolarizing_rate, num_qubits=2)

        # build noise model
        nm = NoiseModel()
        nm.add_noise_for_all_qubits(single_dep, ['h', 'u1'])
        nm.add(double_dep, ['cx'])

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulation()
        _ = dm_simu.run(TestNoise.circuit, noise_model=nm)

        assert 1

    def test_damping(self):
        # Amplitude damping error
        amp_err = DampingError(amplitude_prob=0.1, phase_prob=0, dissipation_state=0.4)
        # Phase damping error
        phase_err = DampingError(amplitude_prob=0, phase_prob=0.3)
        # Amp + Phase damping error
        amp_phase_err = DampingError(amplitude_prob=0.1, phase_prob=0.3, dissipation_state=0.5)

        # build noise model
        nm = NoiseModel()
        nm.add_noise_for_all_qubits(amp_err, ['h', 'u1'])
        nm.add(phase_err, ['y'], 1)
        nm.add(amp_phase_err, ['x'], 1)

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulation()
        _ = dm_simu.run(TestNoise.circuit, noise_model=nm)

        assert 1

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
        nm.add_readout_error(single_readout, 4)
        nm.add_readout_error(single_readout, [1, 3])
        nm.add_readout_error(double_readout, [0, 2])

        # Build measured circuit
        cir = Circuit(5)
        H | cir(0)
        CX | cir([0, 1])
        CX | cir([1, 2])
        CX | cir([2, 3])
        CX | cir([3, 4])
        # Measure | cir

        # Using Density Matrix Simulator to simulate
        dm_simu = DensityMatrixSimulation(accumulated_mode=True)
        _ = dm_simu.run(cir, noise_model=nm)
        print(dm_simu.sample(100))

        assert 1


if __name__ == "__main__":
    unittest.main()
