#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/20 上午10:37
# @Author  : Kaiqi Li
# @File    : unit_test

import os
import unittest
import numpy as np

import QuICT.ops.linalg.cpu_calculator as CPUCalculator
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.unitary import UnitarySimulator
from QuICT.algorithm.qft import QFT


if os.environ.get("test_with_gpu", False):
    import cupy as cp
    import QuICT.ops.linalg.gpu_calculator as GPUCalculator


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
        assert np.allclose(np_result, gpu_result)

    def test_tensor(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        B = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)

        np_result = np.kron(A, B)
        gpu_result = GPUCalculator.tensor(A, B, gpu_out=True)
        assert np.allclose(np_result, gpu_result)

    def test_vector_permutation(self):
        A = np.random.random(1 << (TestGPULinalg.seed * 2)).astype(np.complex64)

        # changeInput = False
        mapping = list(range(TestGPULinalg.seed * 2))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.VectorPermutation(A, mapping, changeInput=False)
        gpu_result = GPUCalculator.VectorPermutation(A, mapping, changeInput=False, gpu_out=True)
        assert np.allclose(cpu_result, gpu_result)

        # changeInput = True
        cpu_result_in_place = A.copy()
        gpu_result_in_place = A.copy()

        CPUCalculator.VectorPermutation(cpu_result_in_place, mapping, changeInput=True)
        gpu_result = GPUCalculator.VectorPermutation(gpu_result_in_place, mapping, changeInput=True, gpu_out=True)
        assert np.allclose(cpu_result, cpu_result_in_place)
        assert np.allclose(gpu_result, gpu_result_in_place)

    def test_matrix_permutation(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)

        mapping = list(range(TestGPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.MatrixPermutation(A, mapping, changeInput=False)
        gpu_result = GPUCalculator.MatrixPermutation(A, mapping, changeInput=False, gpu_out=True)
        assert np.allclose(cpu_result, gpu_result)

    def test_matrix_tensorI(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        n, m = 2, 3

        I_N = np.identity(n)
        I_M = np.identity(m)
        np_result = np.kron(np.kron(I_N, A), I_M)
        gpu_result = GPUCalculator.MatrixTensorI(A, n, m, gpu_out=True)
        assert np.allclose(np_result, gpu_result)

    def test_matrix_dot_vector(self):
        qubit_num = 10
        circuit = Circuit(qubit_num)
        QFT(qubit_num) | circuit

        vec = cp.zeros((1 << qubit_num, ), dtype=np.complex64)
        vec.put(0, np.complex64(1))
        vec = GPUCalculator.matrix_dot_vector(
            vec,
            qubit_num,
            circuit.matrix(),
            list(range(9, -1, -1)),
            sync=True
        )

        sim = UnitarySimulator("GPU")
        sv = sim.run(circuit.matrix())

        assert np.allclose(vec, sv)


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
        assert np.allclose(cpu_result, np_result)

    def test_tensor_cpu(self):
        np_result = np.kron(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)
        cpu_result = CPUCalculator.tensor(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)
        self.assertTrue((np_result == cpu_result).all())

    def test_multiply_cpu(self):
        np_result = np.multiply(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)
        cpu_result = CPUCalculator.multiply(TestCPULinalg.matrix_A, TestCPULinalg.matrix_B)

        self.assertTrue(np.allclose(np_result, cpu_result))

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
        self.assertTrue(np.isclose(np.sum(cpu_result), np.sum(TestCPULinalg.vector), atol=1e-04))

    def test_matrixpermutation_cpu(self):
        mapping = list(range(TestCPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.MatrixPermutation(TestCPULinalg.matrix_A, mapping)
        self.assertTrue(np.isclose(np.sum(cpu_result), np.sum(TestCPULinalg.matrix_A), atol=1e-04))

    def test_matrix_dot_vector(self):
        from QuICT.ops.gate_kernel.cpu import matrix_dot_vector

        qubit_num = 10
        circuit = Circuit(qubit_num)
        QFT(qubit_num) | circuit

        vec = np.zeros((1 << qubit_num, ), dtype=np.complex128)
        vec[0] = np.complex128(1)
        matrix_dot_vector(
            vec,
            circuit.matrix(),
            np.array(list(range(10)))
        )

        sim = UnitarySimulator("CPU")
        sv = sim.run(circuit.matrix())

        self.assertTrue(np.allclose(vec, sv))

    def test_measure_gate_apply(self):
        from QuICT.ops.gate_kernel.cpu import measure_gate_apply

        qubit_num = 10
        circuit = Circuit(qubit_num)
        QFT(qubit_num) | circuit

        vec = np.zeros((1 << qubit_num, ), dtype=np.complex128)
        vec[0] = np.complex128(1)
        vec = measure_gate_apply(
            qubit_num,
            vec
        )

        self.assertTrue(np.allclose(vec, 0))


if __name__ == "__main__":
    unittest.main()
