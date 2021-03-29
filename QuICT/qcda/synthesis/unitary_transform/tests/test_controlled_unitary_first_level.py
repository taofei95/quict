# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from scipy.linalg import block_diag
from scipy.stats import unitary_group
from ..controlled_unitary import QuantumShannonDecompose
from ...uniformly_gate import uniformlyRz
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_controlled_unitary_first_level():  # Only test the first decomposition
    rnd = 10
    for _ in range(rnd):
        qubit_num = 6
        dim = 1 << qubit_num
        u1 = unitary_group.rvs(dim // 2)
        u2 = unitary_group.rvs(dim // 2)
        v, d, w = QuantumShannonDecompose.decompose(u1, u2)
        angle_list = []
        for i in range(d.shape[0]):
            s = d[i, i]
            theta = -2 * np.log(s) / 1j
            angle_list.append(theta)

        gates = uniformlyRz(angle_list=angle_list) \
            .build_gate(mapping=[(i + 1) % qubit_num for i in range(qubit_num)])
        circuit = Circuit(qubit_num)
        circuit.extend(gates)
        mat = SyntheticalUnitary.run(circuit)
        assert np.allclose(mat, block_diag(d, d.conj().T))
        assert np.allclose(block_diag(u1, u2), block_diag(v, v) @ mat @ block_diag(w, w))
