#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/23 4:30
# @Author  : Han Yu
# @File    : model_unit_test.py

import pytest
import random

import numpy as np

from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary

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
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_QFT():
    max_test = 5
    every_round = 20
    for i in range(4, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            X | circuit
            QFT | circuit
            IQFT | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_RZZ():
    max_test = 5
    every_round = 20
    for i in range(4, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            X | circuit
            ran = random.random() * np.pi
            RZZ(ran) | circuit([0, 1])
            RZZ(-ran) | circuit([0, 1])
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_CU1():
    max_test = 6
    every_round = 20
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran = random.random() * np.pi
            X | circuit
            CU1(ran) | circuit([0, 1])
            CU1(-ran) | circuit([0, 1])
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_CU3():
    max_test = 6
    every_round = 20
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran1 = random.random() * np.pi
            ran2 = random.random() * np.pi
            ran3 = random.random() * np.pi
            X | circuit
            CU3([ran1, ran2, ran3]) | circuit([0, 1])
            CU3([ran1, np.pi - ran3, np.pi - ran2]) | circuit([0, 1])
            # U3([ran1, ran2, ran3]) | circuit
            # U3([ran1, np.pi - ran3, np.pi - ran2]) | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_Fredkin():
    max_test = 6
    every_round = 20
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            X | circuit
            Fredkin | circuit
            Fredkin | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_CCX():
    max_test = 6
    every_round = 20
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            X | circuit
            CCX_Decompose | circuit
            CCX_Decompose | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_CRz():
    max_test = 6
    every_round = 20
    for i in range(3, max_test + 1):
        for _ in range(every_round):
            circuit = Circuit(i)
            ran = random.random() * np.pi
            X | circuit
            CRz_Decompose(ran) | circuit
            CRz_Decompose(-ran) | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
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
            CCRz(-ran) | circuit
            X | circuit
            unitary = SyntheticalUnitary.run(circuit)
            if (abs(abs(unitary - np.identity((1 << i), dtype=np.complex))) > 1e-10).any():
                assert 0
    assert 1

def test_gate_name():
    circuit = Circuit(5)
    X % "AA" | circuit
    X % 1 | circuit(1)
    CX | circuit([1, 2])
    circuit.print_infomation()
    assert 1

if __name__ == "__main__":
    # pytest.main(["./_unit_test.py", "./circuit_unit_test.py", "./gate_unit_test.py", "./qubit_unit_test.py"])
    pytest.main(["./gate_unit_test.py"])
