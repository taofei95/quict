# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_csd():
    rnd = 20
    for _ in range(rnd):
        n = 6
        circuit = Circuit(n)
        circuit.random_append(rand_size=20)
        mat = SyntheticalUnitary.run(circuit)
        mat_size = mat.shape[0]
        u, cs, v_dagger = cossin(mat, mat_size // 2, mat_size // 2)
        mult_mat = u @ cs @ v_dagger
        assert np.allclose(mat, mult_mat)
