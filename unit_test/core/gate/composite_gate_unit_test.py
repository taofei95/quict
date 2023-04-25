#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/18 12:03 上午
# @Author  : Li Kaiqi
# @File    : composite_gate_unit_test
import unittest

from QuICT.core import Circuit
from QuICT.core.gate import *


class TestCompositeGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Composite Gate unit test start!")
        cls.qubits = 5

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Composite Gate unit test finished!")

    def test_compositegate_build(self):
        cgate = CompositeGate()
        H | cgate(1)
        U1(0) | cgate(2)
        CH & [0, 1] | cgate
        CX | cgate([3, 4])
        CU3(1, 0, 0) | cgate([0, 4])
        CCRz(1) | cgate([0, 1, 2])
        QFT(3) | cgate

        assert cgate.size() == 12 and cgate.depth() == 9
        assert cgate.count_1qubit_gate() == 5 and cgate.count_2qubit_gate() == 6

    def test_compositegate_matrix(self):
        test_gate = CompositeGate()
        H | test_gate(0)
        H | test_gate(1)
        H | test_gate(2)
        CX | test_gate([0, 1])
        CX | test_gate([0, 2])

        matrix = test_gate.matrix()
        assert matrix.shape == (1 << 3, 1 << 3)

    def test_compositegate_merge(self):
        # composite gate | composite gate
        cgate1 = CompositeGate()
        X | cgate1(0)
        Y | cgate1(1)
        Z | cgate1(2)
        cgate2 = CompositeGate()
        CX | cgate2([0, 1])
        CY | cgate2([1, 2])
        CZ | cgate2([2, 3])
        cgate1 | cgate2
        assert cgate2.size() == 6

        # composite gate ^ composite gate
        cgate1 ^ cgate2
        assert cgate2.size() == 9
        cgate2 ^ cgate2
        assert cgate2.size() == 18

        # composite gate | circuit
        cir = Circuit(5)
        cgate2 | cir
        assert cir.size() == 18

        cgate3 = CompositeGate()
        with cgate3:
            SX & 0
            SY & 1
            SW & 2
        cgate3 | cir
        assert cir.size() == 21

    def test_QFT_gate(self):
        qft = QFT(TestCompositeGate.qubits)
        qft.count_gate_by_gatetype(GateType.h) == 5
        assert qft.size() == 15
        assert qft.depth() == 9
        
        iqft = IQFT(TestCompositeGate.qubits)
        qft.count_gate_by_gatetype(GateType.h) == 5
        assert iqft.size() == 15
        assert iqft.depth() == 9
        
        assert qft.inverse() != qft

if __name__ == "__main__":
    unittest.main()
