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
        mat = f2_half_gaussian_elimination(mat_)
        t = min(n, m)
        for i in range(t):
            for j in range(i):
                assert not mat[i, j]


def test_f2_inverse():
    rnd = 2000
    for _ in range(rnd):
        n = random.randint(2, 200)
        mat_ = f2_random_invertible_matrix_gen(n)
        mat = mat_.copy()
        rk = f2_rank(mat)
        assert rk == n
        mat_inv = f2_inverse(mat)
        assert np.allclose(mat, mat_)
        prod = f2_prod(mat_inv, mat)
        eye = np.eye(n, dtype=bool)
        assert np.allclose(prod, eye)
