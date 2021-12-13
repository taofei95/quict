#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/23 4:30
# @Author  : Han Yu
# @File    : model_unit_test.py

import random

import numpy as np
import pytest

from QuICT.algorithm import Amplitude, SyntheticalUnitary
from QuICT.core import *


def test_permMulDetail():
    max_test = 6
    every_round = 20
    for i in range(4, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            X | circuit
            ControlPermMulDetail([2, 5]) | circuit
            ControlPermMulDetail([2, 5]).inverse() | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex128))) > 1e-10).any():
                assert 0
    assert 1


def test_CCRz():
    max_test = 6
    every_round = 20
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            X | circuit
            ran = random.random() * np.pi
            CCRz(ran) | circuit
            # amplitude = Amplitude.run(circuit)
            # print(amplitude)
            CCRz(-ran) | circuit
            # amplitude = Amplitude.run(circuit)
            # print(amplitude)
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex128))) > 1e-10).any():
                assert 0
            assert 1
    assert 1


def test_gate_name():
    circuit = Circuit(5)
    X(name="XX") | circuit
    X(name=1) | circuit(1)
    CX | circuit([1, 2])

    circuit.print_information()
    assert 1


def test_fSim():
    max_test = 6
    every_round = 20
    for i in range(2, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran1 = random.random() * np.pi
            ran2 = random.random() * np.pi
            # X | circuit(0)
            FSim([ran1, ran2]) | circuit
            # X | circuit(0)
            FSim([-ran1, -ran2]) | circuit
            amplitude = Amplitude.run(circuit)
            amplitudes = np.zeros(1 << i)
            amplitudes[0] = 1
            # print(amplitude)
            if (abs(abs(amplitude - amplitudes)) > 1e-10).any():
                print(amplitude)
                assert 0
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex128))) > 1e-10).any():
                # print(unitary)
                assert 0
    assert 1


def test_Rxx():
    max_test = 6
    every_round = 20
    for i in range(2, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran = random.random() * np.pi
            # X | circuit(0)
            Rxx(ran) | circuit
            # X | circuit(0)
            Rxx(-ran) | circuit
            amplitude = Amplitude.run(circuit)
            amplitudes = np.zeros(1 << i)
            amplitudes[0] = 1
            # print(amplitude)
            if (abs(abs(amplitude - amplitudes)) > 1e-10).any():
                print(amplitude)
                assert 0
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex128))) > 1e-10).any():
                # print(unitary)
                assert 0
    assert 1


def test_Ryy():
    max_test = 6
    every_round = 20
    for i in range(2, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran = random.random() * np.pi
            # X | circuit(0)
            Ryy(ran) | circuit
            # X | circuit(0)
            Ryy(-ran) | circuit
            amplitude = Amplitude.run(circuit)
            amplitudes = np.zeros(1 << i)
            amplitudes[0] = 1
            # print(amplitude)
            if (abs(abs(amplitude - amplitudes)) > 1e-10).any():
                print(amplitude)
                assert 0
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex128))) > 1e-10).any():
                # print(unitary)
                assert 0
    assert 1


def test_Rzz():
    max_test = 6
    every_round = 20
    for i in range(2, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran = random.random() * np.pi
            # X | circuit(0)
            Rzz(ran) | circuit
            # X | circuit(0)
            Rzz(-ran) | circuit
            amplitude = Amplitude.run(circuit)
            amplitudes = np.zeros(1 << i)
            amplitudes[0] = 1
            # print(amplitude)
            if (abs(abs(amplitude - amplitudes)) > 1e-10).any():
                print(amplitude)
                assert 0
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex128))) > 1e-10).any():
                # print(unitary)
                assert 0
    assert 1


def test_CCX():
    max_test = 3
    every_round = 1
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            CCX | circuit
            amplitude1 = Amplitude.run(circuit)
            circuit.clear()
            (CCX & [0, 1, 2]).build_gate() | circuit
            amplitude2 = Amplitude.run(circuit)
            assert np.allclose(amplitude1, amplitude2)


if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    pytest.main(["./gate_unit_test.py", "./new_gate_builder_unit_test.py"])
