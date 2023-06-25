# noinspection PyUnresolvedReferences
import numpy as np
from scipy.linalg import cossin
from scipy.linalg import block_diag

from QuICT.core import Circuit
from QuICT.core.gate import GateType
from QuICT.core.gate.backend import UniformlyRotation


def test_csd():
    n = 4
    circuit = Circuit(n)
    circuit.random_append(rand_size=20)
    mat1 = circuit.matrix()
    mat_size = mat1.shape[0]
    _, cs, _ = cossin(mat1, mat_size // 2, mat_size // 2)
    u, angle_list, v_dagger = cossin(mat1, mat_size // 2, mat_size // 2, separate=True)
    angle_list *= 2  # Ry gate use its angle as theta/2

    circuit_2 = Circuit(n)
    URy = UniformlyRotation(GateType.ry)
    gates = URy.execute(angle_list)
    gates & [(i + 1) % n for i in range(n)]
    circuit_2.extend(gates)
    mat2 = gates.matrix()
    assert np.allclose(cs, mat2)
    assert np.allclose(block_diag(u[0], u[1]) @ mat2 @ block_diag(v_dagger[0], v_dagger[1]), mat1)
