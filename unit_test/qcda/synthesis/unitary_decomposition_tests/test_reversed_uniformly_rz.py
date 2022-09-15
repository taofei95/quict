# noinspection PyUnresolvedReferences
import numpy as np

from QuICT.qcda.synthesis.uniformly_gate import UniformlyRotation
from QuICT.core.gate import GateType


def test_reversed_uniformly_rz():
    rnd = 10
    for _ in range(rnd):
        controlled_cnt = 5
        qubit_cnt = controlled_cnt + 1
        angle_cnt = 1 << controlled_cnt
        angle_list = [np.random.uniform(low=0, high=np.pi) for _ in range(angle_cnt)]
        URz = UniformlyRotation(GateType.rz)
        gates = URz.execute(angle_list)
        gates & [(i + 1) % qubit_cnt for i in range(qubit_cnt)]
        mat = gates.matrix()
        for i in range(angle_cnt):
            assert np.isclose(np.exp(-1j * angle_list[i] / 2), mat[i, i])
            assert np.isclose(np.exp(1j * angle_list[i] / 2), mat[i + angle_cnt, i + angle_cnt])