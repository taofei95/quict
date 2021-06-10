# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from QuICT.qcda.synthesis.uniformly_gate import UniformlyRy
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_reversed_uniformly_ry():
    rnd = 10
    for _ in range(rnd):
        controlled_cnt = 5
        qubit_num = controlled_cnt + 1
        angle_cnt = 1 << controlled_cnt
        angle_list = [np.random.uniform(low=0, high=np.pi) for _ in range(angle_cnt)]
        gates = UniformlyRy.execute(
            angle_list=angle_list,
            mapping=[(i + 1) % qubit_num for i in range(qubit_num)]
        )
        mat = gates.matrix()
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
