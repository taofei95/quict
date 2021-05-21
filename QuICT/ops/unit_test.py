import os
import unittest
import numpy as np

import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.ops.linalg.gpu_calculator as GPUCalculator


# @unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGPULinalg(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print(f"The GPU linalg unit test start!")
        cls.seed = np.random.randint(3, 7)

    @classmethod
    def tearDownClass(cls) -> None:
        print(f"The GPU linalg unit test finished!")

    def test_dot(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)
        B = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)

        np_result = np.dot(A, B)

        gpu_result = GPUCalculator.dot(A, B, gpu_out=True)
        self.assertTrue((np_result == gpu_result).all())

    def test_tensor(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)
        B = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)

        np_result = np.kron(A, B)

        gpu_result = GPUCalculator.tensor(A, B, gpu_out=True)
        self.assertTrue((np_result == gpu_result).all())

    def test_vector_permutation(self):
        A = np.random.random(1 << (TestGPULinalg.seed * 2)).astype(np.complex128)

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
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)

        mapping = list(range(TestGPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.MatrixPermutation(A, mapping, changeInput=False)
        gpu_result = GPUCalculator.MatrixPermutation(A, mapping, changeInput=False, gpu_out=True)
        self.assertTrue((cpu_result == gpu_result).all())

    def test_matrix_tensorI(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)
        n, m = 2, 3

        I_N = np.identity(n)
        I_M = np.identity(m)
        np_result = np.kron(np.kron(I_N, A), I_M)
        gpu_result = GPUCalculator.MatrixTensorI(A, n, m, gpu_out=True)
        self.assertTrue((np_result == gpu_result).all())

    def test_vector_dot(self):
        A = np.random.random((1 << TestGPULinalg.seed, 1 << TestGPULinalg.seed)).astype(np.complex128)
        V = np.random.random(1 << (TestGPULinalg.seed * 2)).astype(np.complex128)
        mapping = list(range(TestGPULinalg.seed))[::-1]
        mapping = np.array(mapping)
        np.random.shuffle(mapping)

        cpu_result = CPUCalculator.vectordot(A, V, mapping)

        gpu_result = GPUCalculator.vectordot(A, V, mapping, gpu_out=True)
        self.assertTrue(np.allclose(cpu_result, gpu_result))

    def test_small_mat_large_vec_cuda_kernel(self):
        qubit_num = np.random.randint(3, 10)
        affect_num = np.random.randint(1, min(qubit_num, 4))
        affect_args_ = np.random.choice(np.arange(qubit_num), affect_num, False)
        small_mat_ = np.random.rand(1 << affect_num, 1 << affect_num) + \
                     np.random.rand(1 << affect_num, 1 << affect_num) * 1.0j
        large_vec_ = np.random.rand(1 << qubit_num) + np.random.rand(1 << qubit_num) * 1.0j
        small_mat = GPUCalculator.htod(small_mat_)
        large_vec = GPUCalculator.htod(large_vec_)
        affect_args = GPUCalculator.htod(affect_args_)
        result_expected = GPUCalculator.vector_dot_refined(small_mat, large_vec, affect_args)

        result = GPUCalculator.htod(large_vec_)
        reverse_mapping = [i for i in range(qubit_num)]
        reverse_mapping.reverse()
        reverse_mapping = np.array(reverse_mapping)
        result = GPUCalculator.VectorPermutation(result, reverse_mapping)
        GPUCalculator.vector_dot_cuda(small_mat, result, affect_args)
        result = GPUCalculator.VectorPermutation(result, reverse_mapping)
        self.assertTrue(np.allclose(result, result_expected))


if __name__ == "__main__":
    unittest.main()
