# noinspection PyUnresolvedReferences
import numpy as np
from scipy.stats import unitary_group
from QuICT.qcda.synthesis.unitary_transform.controlled_unitary import quantum_shannon_decompose
from QuICT.core import *


def test_qsd():
    rnd = 10
    for _ in range(rnd):
        n = 6
        dim = 1 << n
        u1 = unitary_group.rvs(dim)
        u2 = unitary_group.rvs(dim)
        v, d, w = quantum_shannon_decompose(u1, u2)
        d_dagger = d.conj()
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if i == j:
                    continue
                assert np.isclose(d[i, j], 0)
        assert np.allclose(u1, v @ d @ w)
        assert np.allclose(u2, v @ d_dagger @ w)
