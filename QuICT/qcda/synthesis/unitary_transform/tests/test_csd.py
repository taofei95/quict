# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from scipy.linalg import block_diag
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary
from ...uniformly_gate.uniformly_rotation import uniformlyRy


def test_csd():
    rnd = 20
    for _ in range(rnd):
        n = 6
        circuit = Circuit(n)
        circuit.random_append(rand_size=20)
        mat1 = SyntheticalUnitary.run(circuit)
        mat_size = mat1.shape[0]
        _, cs, _ = cossin(mat1, mat_size // 2, mat_size // 2)
        u, angle_list, v_dagger = cossin(mat1, mat_size // 2, mat_size // 2, separate=True)
        angle_list *= 2  # Ry gate use its angle as theta/2

        circuit_2 = Circuit(n)
        gates = uniformlyRy(angle_list=angle_list).build_gate(mapping=[(i + 1) % n for i in range(n)])
        circuit_2.extend(gates)
        mat2 = SyntheticalUnitary.run(circuit_2)
        assert np.allclose(cs, mat2)
        assert np.allclose(block_diag(u[0], u[1]) @ mat2 @ block_diag(v_dagger[0], v_dagger[1]), mat1)
