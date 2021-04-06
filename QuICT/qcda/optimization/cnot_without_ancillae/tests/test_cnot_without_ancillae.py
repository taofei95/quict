import random
import numpy as np

from ..cnot_without_ancillae import CnotWithoutAncillae
from ..block_ldu_decompose import BlockLDUDecompose

from .utility import *
from ..utility import *


def test_triangular_matrix_run():
    rnd = 200
    for _ in range(rnd):
        n = random.randint(2, 200)
        s_size = n // 2
        t_size = n - s_size
        eye = np.eye(n, dtype=bool)
        mat = f2_random_invertible_matrix_gen(n)

        mat_cpy: np.ndarray = mat.copy()

        lower, _, upper, _ = BlockLDUDecompose.run(mat)

        assert np.allclose(mat, mat_cpy)

        # Test matrix construction
        for i in range(n):
            for j in range(n):
                if i < j:
                    assert not lower[i, j]
                if i > j:
                    assert not upper[i, j]

        # Test parallel row elimination
        lower_parallel_elimination = \
            CnotWithoutAncillae.triangular_matrix_run(mat=lower, is_lower_triangular=True)
        upper_parallel_elimination = \
            CnotWithoutAncillae.triangular_matrix_run(mat=upper, is_lower_triangular=False)

        # Correctness of elimination
        for elimination_level in lower_parallel_elimination:
            for c, t in elimination_level:
                assert c < s_size
                assert t >= s_size
                lower[t, :] ^= lower[c, :]
        assert np.allclose(lower, eye)
        for elimination_level in upper_parallel_elimination:
            for c, t in elimination_level:
                assert c >= s_size
                assert t < s_size
                upper[t, :] ^= upper[c, :]
        assert np.allclose(upper, eye)

        # No overlapping
        for elimination_level in lower_parallel_elimination:
            row_index_set = set([])
            for c, t in elimination_level:
                assert c not in row_index_set
                assert t not in row_index_set
                row_index_set.add(c)
                row_index_set.add(t)
        for elimination_level in upper_parallel_elimination:
            row_index_set = set([])
            for c, t in elimination_level:
                assert c not in row_index_set
                assert t not in row_index_set
                row_index_set.add(c)
                row_index_set.add(t)


def test_matrix_run():
    rnd = 200
    for _ in range(rnd):
        n = random.randint(2, 200)
        s_size = n // 2
        eye = np.eye(n, dtype=bool)
        mat = f2_random_invertible_matrix_gen(n)

        mat_cpy: np.ndarray = mat.copy()

        parallel_elimination = CnotWithoutAncillae.matrix_run(mat)
        assert np.allclose(mat, mat_cpy)

        # Test elimination correctness
        for elimination_level in parallel_elimination:
            for c, t in elimination_level:
                mat[t, :] ^= mat[c, :]
        assert np.allclose(mat, eye)

        # No overlapping
        for elimination_level in parallel_elimination:
            level_set = set([])
            for c, t in elimination_level:
                assert c not in level_set
                assert t not in level_set
                level_set.add(c)
                level_set.add(t)


def test_small_matrix_run():
    rnd = 200
    for _ in range(rnd):
        mat = np.eye(2, dtype=bool)
        for _ in range(10):
            x = [0, 1]
            random.shuffle(x)
            i, j = x[0], x[1]
            mat[j, :] ^= mat[i, :]

        parallel_elimination = CnotWithoutAncillae.small_matrix_run(mat)
        for elimination_level in parallel_elimination:
            for c, t in elimination_level:
                mat[t, :] ^= mat[c, :]
        assert np.allclose(mat, np.eye(2, dtype=bool))


def test_remapping_run():
    rnd = 200
    for _ in range(rnd):
        n = random.randint(2, 200)
        s_size = n // 2
        t_size = n - s_size
        eye = np.eye(n, dtype=bool)
        mat = f2_random_invertible_matrix_gen(n)

        mat_cpy: np.ndarray = mat.copy()

        _, _, _, remapping = BlockLDUDecompose.run(mat)
        assert np.allclose(mat_cpy, mat)

        remapped_mat: np.ndarray = eye[remapping].copy()

        parallel_elimination = CnotWithoutAncillae.remapping_run(remapping)
        assert len(parallel_elimination) == 3
        for elimination_level in parallel_elimination:
            for c, t in elimination_level:
                remapped_mat[t, :] ^= remapped_mat[c, :]
        assert np.allclose(remapped_mat, eye)


def test_recursion_first_level():
    rnd = 200
    for _ in range(rnd):
        n = random.randint(2, 200)
        s_size = n // 2
        t_size = n - s_size
        mat = f2_random_invertible_matrix_gen(n)
        l_, d_, u_, remapping = BlockLDUDecompose.run(mat)
        p_: np.ndarray = np.eye(n, dtype=bool)[remapping]

        mult_result = f2_matmul(p_, l_)
        mult_result = f2_matmul(mult_result, d_)
        mult_result = f2_matmul(mult_result, u_)
        assert np.allclose(mult_result, mat)

        p_e = CnotWithoutAncillae.remapping_run(remapping)
        p_e.extend(CnotWithoutAncillae.triangular_matrix_run(l_, is_lower_triangular=True))

        for elimination_level in p_e:
            for c, t in elimination_level:
                mat[t, :] ^= mat[c, :]

        d_inv = f2_inverse(d_)
        mat = f2_matmul(d_inv, mat)

        p_e = CnotWithoutAncillae.triangular_matrix_run(u_, is_lower_triangular=False)
        for level in p_e:
            for c, t in level:
                mat[t, :] ^= mat[c, :]

        assert np.allclose(mat, np.eye(n, dtype=bool))
