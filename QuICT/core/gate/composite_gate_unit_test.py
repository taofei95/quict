#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/28 12:03 上午
# @Author  : Han Yu
# @File    : composite_gate_unit_test
import unittest

from QuICT.core import Circuit
from QuICT.core.gate import *


class TestCompositeGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Composite Gate unit test start!")
        cls.qubits = 5
        cls.based_composite_gate = CompositeGate(cls.qubits)

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Composite Gate unit test finished!")
        del cls.based_composite_gate

    def _gate_attr_test(self, cgate):
        return (
            cgate.size() == 5 and
            cgate.count_1qubit_gate() == 2 and
            cgate.count_2qubit_gate() == 2 and
            cgate.depth() == 3
        )

    def test_compositegate_build(self):
        # Composite Gate Initial
        cgate = TestCompositeGate.based_composite_gate
        assert cgate.width() == TestCompositeGate.qubits

        H | cgate(1)
        U1(0) | cgate(2)
        CX | cgate([3, 4])
        CU3(1, 0, 0) | cgate([0, 4])
        CCRz(1) | cgate([0, 1, 2])
        assert self._gate_attr_test(cgate)

    def test_compositegate_matrix(self):
        test_gate = CompositeGate(3)
        H | test_gate(0)
        H | test_gate(1)
        H | test_gate(2)
        CX | test_gate([0, 1])
        CX | test_gate([0, 2])
        CCRz | test_gate([0, 1, 2])

        matrix = test_gate.matrix()
        assert matrix.shape == (1 << 3, 1 << 3)

    def test_compositegate_operation(self):
        # composite gate | composite gate
        ncgate = CompositeGate(TestCompositeGate.qubits)
        TestCompositeGate.based_composite_gate | ncgate
        assert self._gate_attr_test(ncgate)

        # composite gate ^ composite gate
        TestCompositeGate.based_composite_gate ^ ncgate
        assert ncgate.size() == 10

        # composite gate | circuit
        cir = Circuit(5)
        ncgate | cir
        assert cir.size() == 10


if __name__ == "__main__":
    unittest.main()
