from scipy.stats import unitary_group
from ..unitary_transform import UTrans
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_unitary_transform_base_2():
    rnd = 10
    for _ in range(rnd):
        qubit_num = np.random.randint(1, 7)
        mat1 = unitary_group.rvs(1 << qubit_num)
        synthesized_circuit = Circuit(qubit_num)
        mat1_cpy = mat1.copy()
        gates = UTrans(mat1, recursive_basis=2).build_gate()
        synthesized_circuit.extend(gates)
        mat2 = SyntheticalUnitary.run(synthesized_circuit)
        assert np.allclose(mat1, mat1_cpy)
        assert np.isclose(np.linalg.det(mat1), np.linalg.det(mat2))
        assert np.allclose(mat1, mat2)
