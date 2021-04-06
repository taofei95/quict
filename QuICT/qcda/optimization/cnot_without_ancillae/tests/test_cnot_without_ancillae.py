import random
import numpy as np

from ..cnot_without_ancillae import CnotWithoutAncillae
from ..block_ldu_decompose import BlockLDUDecompose


def test_triangular_matrix_run():
    rnd = 200
    for _ in range(rnd):
        n = random.randint(2, 200)
        s_size = n // 2
        t_size = n - s_size
        eye = np.eye(n, dtype=bool)
        mat = np.eye(n, dtype=bool)

        # Prepare matrices
        rg_lst = list(range(n))
        for _ in range(30 * n):
            x = random.sample(rg_lst, 2)
            i, j = x[0], x[1]
            mat[j, :] ^= mat[i, :]

        lower, _, upper, _ = BlockLDUDecompose.run(mat)

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
