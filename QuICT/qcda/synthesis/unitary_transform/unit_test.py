import pytest
import numpy as np
# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from scipy.linalg import block_diag
from scipy.stats import unitary_group

from .controlled_unitary import QuantumShannonDecompose
from .unitary_transform import UTrans

from ..uniformly_gate import uniformlyRz, uniformlyRy

from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


# @pytest.mark.skip(reason="Save debug time")
def test_csd():
    rnd = 10
    for _ in range(rnd):
        n = 6
        circuit = Circuit(n)
        circuit.random_append(rand_size=20)
        mat = SyntheticalUnitary.run(circuit)
        mat_size = mat.shape[0]
        u, cs, v_dagger = cossin(mat, mat_size // 2, mat_size // 2)
        mult_mat = u @ cs @ v_dagger
        assert np.allclose(mat, mult_mat)


# @pytest.mark.skip(reason="Save debug time")
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


# @pytest.mark.skip(reason="Save debug time")
def test_reversed_uniformly_rz():
    rnd = 10
    for _ in range(rnd):
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
            assert np.isclose(np.exp(1J * angle_list[i] / 2), mat[i + angle_cnt, i + angle_cnt])


# @pytest.mark.skip(reason="Save debug time")
def test_reversed_uniformly_ry():
    rnd = 10
    for _ in range(rnd):
        controlled_cnt = 5
        qubit_num = controlled_cnt + 1
        angle_cnt = 1 << controlled_cnt
        angle_list = [np.random.uniform(low=0, high=np.pi) for _ in range(angle_cnt)]
        gates = uniformlyRy(angle_list=angle_list) \
            .build_gate(mapping=[(i + 1) % qubit_num for i in range(qubit_num)])
        circuit_seg = Circuit(qubit_num)
        circuit_seg.extend(gates)
        mat = SyntheticalUnitary.run(circuit_seg)
        for i in range(angle_cnt):
            for j in range(angle_cnt):
                if i == j:
                    c = np.cos(angle_list[i] / 2)
                    s = np.sin(angle_list[i] / 2)
                    assert np.isclose(c, mat[i, i])
                    assert np.isclose(-s, mat[i, i + angle_cnt])
                    assert np.isclose(s, mat[i + angle_cnt, i])
                    assert np.isclose(c, mat[i + angle_cnt, i + angle_cnt])
                else:
                    assert np.isclose(0, mat[i, j])


# @pytest.mark.skip(reason="Save debug time")
def test_controlled_unitary():  # Only test the first decomposition
    rnd = 10
    for _ in range(rnd):
        n = 5
        dim = 1 << n
        u1 = unitary_group.rvs(dim // 2)
        u2 = unitary_group.rvs(dim // 2)
        _, d, _ = QuantumShannonDecompose.decompose(u1, u2)

        angle_list = []
        for i in range(d.shape[0]):
            s = d[i, i]
            theta = -2 * np.log(s) / 1j
            angle_list.append(theta)

        gates = uniformlyRz(angle_list=angle_list).build_gate(mapping=[(i + 1) % n for i in range(n)])
        circuit = Circuit(n)
        circuit.extend(gates)
        mat = SyntheticalUnitary.run(circuit)
        assert np.allclose(mat, block_diag(d, d.conj().T))


# @pytest.mark.skip(reason="Save debug time")
def test_csd_inner_ry():
    rnd = 10
    for _ in range(rnd):
        controlled_cnt = 5
        qubit_num = controlled_cnt + 1
        mat1 = unitary_group.rvs(1 << qubit_num)
        mat_size = mat1.shape[0]
        _, cs, _ = cossin(mat1, mat_size // 2, mat_size // 2)
        for i in range(mat_size // 2):
            for j in range(mat_size // 2):
                if i != j:
                    assert np.isclose(0, cs[i, j])
                    assert np.isclose(0, cs[i, j + mat_size // 2])
                    assert np.isclose(0, cs[i + mat_size // 2, j])
                    assert np.isclose(0, cs[i + mat_size // 2, j + mat_size // 2])
                else:
                    assert np.isclose(cs[i, i], cs[i + mat_size // 2, i + mat_size // 2])
                    assert np.isclose(cs[i, i + mat_size // 2], -cs[i + mat_size // 2, i])
                    assert np.isclose(abs(cs[i, i]) ** 2 + abs(cs[i, i + mat_size // 2]) ** 2, 1)

        angle_list = []
        angle_cnt = 1 << controlled_cnt
        assert angle_cnt == mat_size // 2
        for i in range(angle_cnt):
            c = cs[i, i]
            s = -cs[i, i + mat_size // 2]
            theta = np.arccos(c)
            if np.isclose(-np.sin(theta), s):
                theta = -theta
            assert np.isclose(np.sin(theta), s)
            assert np.isclose(np.cos(theta), c)
            angle_list.append(theta * 2)
        reversed_ry = uniformlyRy(angle_list=angle_list) \
            .build_gate(mapping=[(i + 1) % qubit_num for i in range(qubit_num)])
        circuit = Circuit(qubit_num)
        circuit.extend(reversed_ry)
        mat2 = SyntheticalUnitary.run(circuit)
        assert np.allclose(mat2, cs)

# def test_unitary_transform_base_1():
#     qubit_num = 2
#     mat1 = unitary_group.rvs(1 << qubit_num)
#     synthesized_circuit = Circuit(qubit_num)
#     gates = UTrans(mat1).build_gate()
#     synthesized_circuit.extend(gates)
#     mat2 = SyntheticalUnitary.run(synthesized_circuit)
#
#     assert np.allclose(mat1, mat2)
