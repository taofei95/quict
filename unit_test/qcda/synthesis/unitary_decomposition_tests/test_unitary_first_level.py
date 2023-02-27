# noinspection PyUnresolvedReferences
import numpy as np
from scipy.linalg import cossin
from scipy.stats import unitary_group

from QuICT.core.gate.backend import UniformlyRotation
from QuICT.core.gate import GateType


def test_unitary_first_level():
    controlled_cnt = 3
    qubit_num = controlled_cnt + 1
    mat1 = unitary_group.rvs(1 << qubit_num)
    mat_size = mat1.shape[0]
    u, cs, v_dagger = cossin(mat1, mat_size // 2, mat_size // 2)
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
        theta *= 2
        angle_list.append(theta)
    URy = UniformlyRotation(GateType.ry)
    reversed_ry = URy.execute(angle_list)
    reversed_ry & [(i + 1) % qubit_num for i in range(qubit_num)]
    mat2 = reversed_ry.matrix()
    assert np.allclose(mat2, cs)
