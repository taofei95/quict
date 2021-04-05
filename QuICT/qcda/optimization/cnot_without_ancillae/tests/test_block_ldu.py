import numpy as np
import random
from ..cnot_without_ancillae import CnotWithoutAncillae
from ..block_ldu_decompose import BlockLDUDecompose
from ..utility import *


def f2_random_invertible_matrix_gen(n) -> np.ndarray:
    mat = np.eye(n, dtype=bool)
    rnd = 20 * n
    rg_lst = list(range(n))
    for _ in range(rnd):
        x = random.sample(rg_lst, 2)
        i = x[0]
        j = x[1]
        mat[i, :] ^= mat[j, :]
    return mat


def test_f2_half_gaussian_elimination():
    rnd = 300
    for _ in range(rnd):
        n = random.randint(1, 200)
        m = random.randint(1, 200)
        mat_ = np.empty(shape=(n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                mat_[i, j] = np.random.choice((True, False))
        mat_cpy = mat_.copy()
        mat = f2_half_gaussian_elimination(mat_)
        assert np.allclose(mat_, mat_cpy)
        t = min(n, m)
        for i in range(t):
            for j in range(i):
                assert not mat[i, j]


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


def test_block_ldu():
    rnd = 300
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
