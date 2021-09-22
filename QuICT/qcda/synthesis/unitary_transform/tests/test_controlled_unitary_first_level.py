# noinspection PyUnresolvedReferences
from scipy.linalg import block_diag
from scipy.stats import unitary_group
from QuICT.qcda.synthesis.unitary_transform.controlled_unitary import quantum_shannon_decompose
from QuICT.qcda.synthesis.uniformly_gate import UniformlyRz
from QuICT.core import *


def test_controlled_unitary_first_level():  # Only test the first decomposition
    rnd = 10
    for _ in range(rnd):
        qubit_num = 6
        dim = 1 << qubit_num
        u1 = unitary_group.rvs(dim // 2)
        u2 = unitary_group.rvs(dim // 2)
        v, d, w = quantum_shannon_decompose(u1, u2)
        angle_list = []
        for i in range(d.shape[0]):
            s = d[i, i]
            theta = -2 * np.log(s) / 1j
            angle_list.append(theta)

        gates = UniformlyRz.execute(
            angle_list=angle_list,
            mapping=[(i + 1) % qubit_num for i in range(qubit_num)]
        )
        mat = gates.matrix()
        assert np.allclose(mat, block_diag(d, d.conj().T))
        assert np.allclose(block_diag(u1, u2), block_diag(v, v) @ mat @ block_diag(w, w))
