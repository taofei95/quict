import numpy as np
import random
from QuICT.qcda.optimization.cnot_without_ancillae.block_ldu_decompose import BlockLDUDecompose
from QuICT.qcda.optimization.cnot_without_ancillae.utility import *
from QuICT.unit_test.qcda.optimization.cnot_without_ancillae_tests.utility import *


def test_f2_half_gaussian_elimination():
    rnd = 500
    for _ in range(rnd):
        m = random.randint(1, 200)
        n = random.randint(1, 200)
        mat_: np.ndarray = np.random.rand(m, n) > 0.5
        mat_cpy: np.ndarray = mat_.copy()
        mat_result = f2_half_gaussian_elimination(mat_)
        assert np.allclose(mat_, mat_cpy)
        last = -1
        for i in range(m):
            j = 0
            while j < n:
                if not mat_result[i, j]:
                    j += 1
                else:
                    break
            assert j > last or j == n
            assert j >= i or j == n
            last = j


def test_f2_rank():
    rnd = 500
    for _ in range(rnd):
        n = random.randint(2, 200)
        expected_rank = random.randint(1, n)
        diag = [True for _ in range(expected_rank)]
        diag.extend([False for _ in range(n - expected_rank)])
        random.shuffle(diag)
        diag = np.array(diag, dtype=bool)
        mat = np.diag(diag)
        rg_lst = [i for i in range(n)]
        for _ in range(20 * n):
            x = random.sample(rg_lst, 2)
            i = x[0]
            j = x[1]
            mat[i, :] ^= mat[j, :]
        rk = f2_rank(mat)
        assert expected_rank == rk


def test_f2_inverse():
    rnd = 1000
    for _ in range(rnd):
        n = random.randint(2, 200)
        mat_ = f2_random_invertible_matrix_gen(n)
        mat_cpy = mat_.copy()
        rk = f2_rank(mat_)
        assert rk == n
        mat_inv = f2_inverse(mat_)
        assert np.allclose(mat_cpy, mat_)
        prod = f2_matmul(mat_inv, mat_)
        eye = np.eye(n, dtype=bool)
        assert np.allclose(prod, eye)


def test_ldu_remapping():
    rnd = 500
    for _ in range(rnd):
        n = random.randint(2, 200)
        mat_ = f2_random_invertible_matrix_gen(n)
        mat_cpy = mat_.copy()
        remapping = BlockLDUDecompose.remapping_select(mat_)
        assert np.allclose(mat_, mat_cpy)
        assert len(remapping) == n
        assert len(set(remapping)) == n
        assert sorted(remapping) == [i for i in range(n)]
        sub_mat: np.ndarray = mat_[remapping][:n // 2, :n // 2]
        assert sub_mat.shape == (n // 2, n // 2)
        assert f2_rank(sub_mat) == n // 2

        for i in range(n):
            assert remapping[remapping[i]] == i


def test_block_ldu():
    rnd = 500
    for _ in range(rnd):
        n = random.randint(2, 200)
        mat_ = f2_random_invertible_matrix_gen(n)
        mat_cpy = mat_.copy()
        l, d, u, remapping = BlockLDUDecompose.run(mat_)
        assert np.allclose(mat_cpy, mat_)
        l_d = f2_matmul(l, d)
        l_d_u = f2_matmul(l_d, u)
        mat = mat_[remapping]

        for i in range(n):
            for j in range(i + 1, n):
                assert not l[i, j]
            for j in range(i):
                assert not u[i, j]
        l_size = n // 2
        for i in range(l_size, n):
            for j in range(l_size):
                assert not d[i, j]
        for i in range(l_size):
            for j in range(l_size, n):
                assert not d[i, j]

        assert np.allclose(mat, l_d_u)
