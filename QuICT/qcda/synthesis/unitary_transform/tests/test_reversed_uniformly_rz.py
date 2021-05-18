# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from ...uniformly_gate import uniformlyRz
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_reversed_uniformly_rz():
    rnd = 10
    for _ in range(rnd):
        controlled_cnt = 5
        qubit_cnt = controlled_cnt + 1
        angle_cnt = 1 << controlled_cnt
        angle_list = [np.random.uniform(low=0, high=np.pi) for _ in range(angle_cnt)]
        gates = uniformlyRz(
            angle_list=angle_list,
            mapping=[(i + 1) % qubit_cnt for i in range(qubit_cnt)]
        )
        mat = gates.matrix()
        for i in range(angle_cnt):
            assert np.isclose(np.exp(-1J * angle_list[i] / 2), mat[i, i])
            assert np.isclose(np.exp(1J * angle_list[i] / 2), mat[i + angle_cnt, i + angle_cnt])
