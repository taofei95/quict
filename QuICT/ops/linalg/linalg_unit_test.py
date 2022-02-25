#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/20 上午10:37
# @Author  : Kaiqi Li
# @File    : unit_test

import os
import unittest
import numpy as np
import cupy as cp

import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.ops.linalg.gpu_calculator as GPUCalculator

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.unitary_simulator import UnitarySimulator


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGPULinalg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The GPU linalg unit test start!")
        cls.seed = np.random.randint(3, 7)

    @classmethod
    def tearDownClass(cls) -> None:
        print("The GPU linalg unit test finished!")

    def test_dot(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        B = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)

        np_result = np.dot(A, B)

        gpu_result = GPUCalculator.dot(A, B, gpu_out=True)
        self.assertTrue((np_result == gpu_result).all())

    def test_tensor(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        B = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)

        np_result = np.kron(A, B)

        gpu_result = GPUCalculator.tensor(A, B, gpu_out=True)
        self.assertTrue((np_result == gpu_result).all())

    def test_vector_permutation(self):
        A = np.random.random(1 << (TestGPULinalg.seed * 2)).astype(np.complex64)

        # changeInput = False
        mapping = list(range(TestGPULinalg.seed * 2))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.VectorPermutation(A, mapping, changeInput=False)
        gpu_result = GPUCalculator.VectorPermutation(A, mapping, changeInput=False, gpu_out=True)
        self.assertTrue((cpu_result == gpu_result).all())

        # changeInput = True
        cpu_result_in_place = A.copy()
        gpu_result_in_place = A.copy()

        CPUCalculator.VectorPermutation(cpu_result_in_place, mapping, changeInput=True)
        gpu_result = GPUCalculator.VectorPermutation(gpu_result_in_place, mapping, changeInput=True, gpu_out=True)
        self.assertTrue((cpu_result == cpu_result_in_place).all())
        self.assertTrue((gpu_result == gpu_result_in_place).all())

    def test_matrix_permutation(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)

        mapping = list(range(TestGPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.MatrixPermutation(A, mapping, changeInput=False)
        gpu_result = GPUCalculator.MatrixPermutation(A, mapping, changeInput=False, gpu_out=True)
        self.assertTrue((cpu_result == gpu_result).all())

    def test_matrix_tensorI(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        n, m = 2, 3

        I_N = np.identity(n)
        I_M = np.identity(m)
        np_result = np.kron(np.kron(I_N, A), I_M)
        gpu_result = GPUCalculator.MatrixTensorI(A, n, m, gpu_out=True)
        self.assertTrue((np_result == gpu_result).all())

    def test_matrix_dot_vector(self):
        qubit_num = 20
        circuit = Circuit(qubit_num)
        QFT.build_gate(qubit_num) | circuit

        anc = cp.zeros((1 << qubit_num, ), dtype=np.complex64)
        vec = cp.zeros((1 << qubit_num, ), dtype=np.complex64)
        vec.put(0, np.complex64(1))

        small_gates = UnitarySimulator.pretreatment(circuit)
        for gate in small_gates:
            GPUCalculator.matrix_dot_vector(
                gate.compute_matrix,
                gate.targets + gate.controls,
                vec,
                qubit_num,
                np.array(gate.cargs + gate.targs, dtype=np.int32),
                auxiliary_vec=anc,
                sync=True
            )
            anc, vec = vec, anc

        assert 1


class TestCPULinalg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The CPU linalg unit test start!")
        cls.seed = np.random.randint(3, 7)

        cls.vector = np.random.random(1 << (cls.seed * 2)).astype(np.complex64)

        cls.matrix_A = np.random.random((1 << cls.seed, 1 << cls.seed)).astype(np.complex64)
        cls.matrix_B = np.random.random((1 << cls.seed, 1 << cls.seed)).astype(np.complex64)

    @classmethod
    def tearDownClass(cls) -> None:
        print("The CPU linalg unit test finished!")

    def test_dot_cpu(self):
        np_result = np.dot(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)

        cpu_result = CPUCalculator.dot(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)
        self.assertTrue((np_result == cpu_result).all())

    def test_tensor_cpu(self):
        np_result = np.kron(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)

        cpu_result = CPUCalculator.tensor(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)
        self.assertTrue((np_result == cpu_result).all())

    def test_MatrixTensorI_cpu(self):
        n, m = 2, 3

        I_N = np.identity(n)
        I_M = np.identity(m)
        np_result = np.kron(np.kron(I_N, TestCPULinalg.matrix_A), I_M)
        cpu_result = CPUCalculator.MatrixTensorI(TestCPULinalg.matrix_A, n, m)
        self.assertTrue((np_result == cpu_result).all())

    def test_vectorpermutation_cpu(self):
        mapping = list(range(TestCPULinalg.seed * 2))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.VectorPermutation(TestCPULinalg.vector, mapping)
        self.assertTrue(np.sum(cpu_result) == np.sum(TestCPULinalg.vector))

    def test_matrixpermutation_cpu(self):
        mapping = list(range(TestCPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.MatrixPermutation(TestCPULinalg.matrix_A, mapping)
        self.assertTrue(np.isclose(np.sum(cpu_result), np.sum(TestCPULinalg.matrix_A), atol=1e-04))


if __name__ == "__main__":
    unittest.main()
