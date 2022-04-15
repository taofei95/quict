#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/23 4:30
# @Author  : Li Kaiqi
# @File    : model_unit_test.py
import numpy as np
import pytest

from QuICT.core import Circuit, Qureg
from QuICT.core.gate import *
from QuICT.core.utils import GateType


def test_gate_build():
    cir = Circuit(10)
    # single qubit gate
    h1 = HGate()
    h1 | cir(1)         # 1
    H | cir             # 11

    # single qubit gate with param
    my_u1 = U1Gate([1])
    my_u1 | cir(2)      # 12
    U1(0) | cir(1)      # 13

    # two qubit gate
    my_CX = CX & [3, 4]
    my_CX | cir         # 14
    CX | cir([3, 4])    # 15

    # two qubit gate with param
    CU3(1, 0, 0) | cir([5, 6])  # 16

    # complexed gate
    CCRz(1) | cir([7, 8, 9])    # 17
    cg_ccrz = CCRz.build_gate()
    cg_ccrz | cir([9, 8, 7])    # 22

    assert len(cir.gates) == 22


def test_gate_name():
    my_gate = PhaseGate()
    assert my_gate.name.split('-')[0] == str(GateType.phase)

    q = Qureg(1)

    cir = Circuit(q)
    my_gate | cir(0)
    assert cir.gates[0].name.split('-')[2] == str(0)


def test_gate_attribute():
    # test single gate
    assert H.is_single() and not CRz.is_single()

    # test control single
    assert CRz.is_control_single() and not H.is_control_single()

    # test Clifford gate
    assert S.is_clifford() and not T.is_clifford()

    # test diagonal gate
    assert S.is_diagonal() and not H.is_diagonal()

    # test Pauli gate
    assert X.is_pauli() and not Rx.is_pauli()

    # test unitary gate diagonal
    dmatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * np.pi / 2)]
    ], dtype=np.complex128)
    bmatrix = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=np.complex128)

    dunitary = Unitary(dmatrix)
    bunitary = Unitary(bmatrix)
    assert dunitary.is_diagonal() and not bunitary.is_diagonal()

    # test special gate
    assert Measure.is_special() and not H.is_special()


if __name__ == "__main__":
    pytest.main(["./gate_unit_test.py"])