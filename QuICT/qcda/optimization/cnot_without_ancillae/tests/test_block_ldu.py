import numpy as np
from ..cnot_without_ancillae import CnotWithoutAncillae
from ..block_ldu_decompose import BlockLDUDecompose


def test_f2_gaussian_elimination():
    rnd = 500
    for _ in range(rnd):
        n = np.random.randint(1, 200)
        m = np.random.randint(1, 200)
        mat_ = np.empty(shape=(n, m), dtype=bool)
        for i in range(n):
            for j in range(m):
                mat_[i, j] = np.random.choice((True, False))
        mat = BlockLDUDecompose.f2_half_gaussian_elimination(mat_)
        t = min(n, m)
        for i in range(t):
            for j in range(i):
                assert not mat[i, j]
