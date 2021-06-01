import os
import unittest
import numpy as np
import cupy as cp

from pprint import pprint
from time import time

import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.ops.linalg.gpu_calculator as GPUCalculator
from QuICT.ops.linalg.calculation_layer import CalculationLayer


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGPULinalg(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"The GPU linalg unit test start!")
        cls.seed = np.random.randint(3, 7)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The GPU linalg unit test finished!")

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

    def test_vector_dot(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        V = np.random.random(1 << (TestGPULinalg.seed * 2)).astype(np.complex64)
        mapping = list(range(TestGPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.vectordot(A, V, mapping)

        gpu_result = GPUCalculator.vectordot(A, V, mapping, gpu_out=True)
        self.assertTrue(np.allclose(cpu_result, gpu_result))

    def test_calculation_layer(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)
        B = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex64)

        based_result = GPUCalculator.dot(A, B, gpu_out=True)

        mempool = cp.get_default_memory_pool()
        before_used_bytes = mempool.used_bytes()

        with CalculationLayer() as CL:
            gpu_A = CL.htod(A)
            gpu_B = CL.htod(B)

            layer_result = CL.dot(gpu_A, gpu_B, gpu_out=True)

        self.assertTrue((based_result==layer_result).all())

        after_used_bytes = mempool.used_bytes()
        # Check for memory release, maybe failure caused by the mulit-process. 
        self.assertEqual(before_used_bytes, after_used_bytes)

    def test_small_mat_large_vec_cuda_kernel(self):
        for _ in range(20):
            qubit_num = np.random.randint(3, 10)
            affect_num = np.random.randint(1, min(qubit_num, 4))
            affect_args_ = np.random.choice(np.arange(qubit_num), affect_num, False)
            small_mat_ = np.random.rand(1 << affect_num, 1 << affect_num) + \
                         np.random.rand(1 << affect_num, 1 << affect_num) * 1.0j
            large_vec_ = np.random.rand(1 << qubit_num) + np.random.rand(1 << qubit_num) * 1.0j

            # # a special test case
            # qubit_num = 3
            # affect_args_ = np.array([2, 0])
            # small_mat_ = np.array([[0.11118886 + 0.97568856j, 0.10932591 + 0.00767598j,
            #                         0.87067396 + 0.25539995j, 0.90238814 + 0.2186802j],
            #                        [0.72960078 + 0.74968579j, 0.21490288 + 0.78974374j,
            #                         0.80567906 + 0.46606459j, 0.47280523 + 0.32588461j],
            #                        [0.76890803 + 0.94829408j, 0.04064005 + 0.43823622j,
            #                         0.01447664 + 0.43654671j, 0.20104284 + 0.67827535j],
            #                        [0.64855955 + 0.09512591j, 0.13268412 + 0.67207301j,
            #                         0.2022993 + 0.99560012j, 0.19299594 + 0.12640645j]])
            # large_vec_ = np.array([0.81312101 + 0.84528757j, 0.42434956 + 0.3850363j,
            #                        0.10807139 + 0.1250964j, 0.41819421 + 0.47631531j,
            #                        0.16126881 + 0.73258657j, 0.45677144 + 0.22861486j,
            #                        0.18473136 + 0.05307219j, 0.652805 + 0.64999251j])

            small_mat = GPUCalculator.htod(small_mat_)
            large_vec = GPUCalculator.htod(large_vec_)
            affect_args = GPUCalculator.htod(affect_args_)
            affect_args_sorted_ = affect_args_.copy()
            affect_args_sorted_.sort()
            affect_args_sorted = GPUCalculator.htod(affect_args_sorted_)
            result_expected = GPUCalculator.vector_dot_refined(small_mat, large_vec, affect_args)
            result_ = large_vec_.copy()
            result = GPUCalculator.htod(result_)

            # # for debug
            # GPUCalculator.vector_dot_cuda_sim(small_mat, result_, affect_args)

            GPUCalculator.vector_dot_cuda(small_mat, result, affect_args, affect_args_sorted)
            result_ = result.get()

            # # print if needed
            # print()
            # pprint(qubit_num)
            # pprint(affect_args_)
            # pprint(small_mat_)
            # pprint(large_vec_)
            # pprint(result_)
            # pprint(result_expected)

            self.assertTrue(np.allclose(result_, result_expected))

    def test_perf_small_mat_large_vec_cuda_kernel(self):
        qubit_num = 25
        affect_num = np.random.randint(2, min(qubit_num, 4))
        affect_args_ = np.random.choice(np.arange(qubit_num), affect_num, False)
        small_mat_ = np.random.rand(1 << affect_num, 1 << affect_num) + \
                     np.random.rand(1 << affect_num, 1 << affect_num) * 1.0j
        large_vec_ = np.random.rand(1 << qubit_num) + np.random.rand(1 << qubit_num) * 1.0j

        small_mat = GPUCalculator.htod(small_mat_)
        large_vec = GPUCalculator.htod(large_vec_)
        affect_args = GPUCalculator.htod(affect_args_)
        affect_args_sorted_ = affect_args_.copy()
        affect_args_sorted_.sort()
        affect_args_sorted = GPUCalculator.htod(affect_args_sorted_)

        rnd = 100
        start_time = time()
        for _ in range(rnd):
            tmp = GPUCalculator.vector_dot_refined(small_mat, large_vec, affect_args, False)
            del tmp
        end_time = time()
        duration_refined = (end_time - start_time) * 1000 / rnd

        # if count jit time
        GPUCalculator.vector_dot_cuda(small_mat, large_vec, affect_args, affect_args_sorted)

        start_time = time()
        for _ in range(rnd):
            GPUCalculator.vector_dot_cuda(small_mat, large_vec, affect_args, affect_args_sorted)
        end_time = time()
        duration_cuda = (end_time - start_time) * 1000 / rnd

        print()
        print(f"qubit_num = {qubit_num}")
        print(f"gate_affect_qubit = {affect_num}")
        print(f"2-permutation dot: {duration_refined:0.4f} ms per op.")
        print(f"cuda: {duration_cuda:0.4f} ms per op.")


if __name__ == "__main__":
    unittest.main()
