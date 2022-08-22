#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 1:20 下午
# @Author  : Li Kaiqi
# @File    : circuit_unit_test.py

import unittest

from QuICT.core import Circuit
from QuICT.core.gate import *


class TestCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Circuit unit test start!")
        cls.qubits = 5
        cls.based_circuit = Circuit(cls.qubits)

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Circuit unit test finished!")
        del cls.based_circuit

    def test_circuit_build(self):
        cir = TestCircuit.based_circuit
        qureg = cir.qubits
        cir.random_append()

        # append supremacy circuit
        cir.supremacy_append()

        # append composite gate
        qft_gate = QFT.build_gate(3)
        qft_gate | cir([0, 1, 3])

        # append gate by qubits/qureg
        S | cir(qureg[0])
        CX | cir(qureg[1, 3])

        assert 1

    def test_circuit_call(self):
        cir = Circuit(TestCircuit.qubits)
        qureg = cir.qubits
        H | cir(1)
        CX | cir([1, 3])
        X | cir(qureg[1])
        CX | cir(qureg([1, 3]))

        assert 1

    def test_sub_circuit(self):
        cir = TestCircuit.based_circuit
        sub_cir_without_remove = cir.sub_circuit(qubit_limit=[0, 1, 2], remove=False)
        assert cir.size() == 76
        assert sub_cir_without_remove.width() == 3

        sub_cir_with_remove = cir.sub_circuit(qubit_limit=[0, 3], remove=True)
        assert cir.size() + sub_cir_with_remove.size() == 76

    def test_circuit_operation(self):
        # append single qubit gate to all qubits
        special_cir = Circuit(TestCircuit.qubits)
        H | special_cir     # 5
        assert special_cir.size() == 5

        # Add gate by qubits
        target_q = special_cir[3]
        S | special_cir(target_q)
        assert special_cir.gates[-1].targs == [3]

        # Add gate by circuit call
        CRz | special_cir([1, 4])
        assert special_cir.gates[-1].targs == [4]

    def test_circuit_matrix_product(self):
        cir = TestCircuit.based_circuit
        mp_gate = CZ & [1, 3]
        mp_data = mp_gate.expand(cir.width())

        assert mp_data.shape == (1 << 5, 1 << 5)

    def test_circuit_remapping(self):
        cir = Circuit(TestCircuit.qubits)
        q1 = cir[1:4]
        assert q1[0] == cir[1]

        cir.remapping(q1, [2, 1, 0])
        assert q1[0] == cir[3]

        q2 = cir[0, 1, 4]
        cir.remapping(q2, [0, 2, 1], circuit_update=True)
        assert q2[0] == cir[0]


if __name__ == "__main__":
    unittest.main()
