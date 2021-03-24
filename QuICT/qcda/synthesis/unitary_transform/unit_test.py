import numpy as np
# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from scipy.stats import unitary_group

from .controlled_unitary import QuantumShannonDecompose

from ..uniformly_gate import uniformlyRz

from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_qsd():
    rnd = 10
    for _ in range(rnd):
        n = 6
        dim = 1 << n
        u1 = unitary_group.rvs(dim)
        u2 = unitary_group.rvs(dim)
        v, d, w = QuantumShannonDecompose.decompose(u1, u2)
        d_dagger = d.conj()
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if i == j:
                    continue
                assert np.isclose(d[i, j], 0)
        assert np.allclose(u1, v @ d @ w)
        assert np.allclose(u2, v @ d_dagger @ w)


def test_reversed_uniformly_rz():
    for _ in range(100):
        controlled_cnt = 5
        qubit_cnt = controlled_cnt + 1
        angle_cnt = 1 << controlled_cnt
        angle_list = [np.random.uniform(low=0, high=np.pi) for _ in range(angle_cnt)]
        gates = uniformlyRz(angle_list=angle_list) \
            .build_gate(mapping=[(i + 1) % qubit_cnt for i in range(qubit_cnt)])
        circuit_seg = Circuit(qubit_cnt)
        circuit_seg.extend(gates)
        mat = SyntheticalUnitary.run(circuit_seg)
        for i in range(angle_cnt):
            assert np.isclose(np.exp(-1J * angle_list[i] / 2), mat[i, i])
