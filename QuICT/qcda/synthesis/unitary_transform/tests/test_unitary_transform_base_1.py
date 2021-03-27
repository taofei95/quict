# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from scipy.stats import unitary_group
from ..unitary_transform import UTrans
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_unitary_transform_base_1():
    qubit_num = 2
    mat1 = unitary_group.rvs(1 << qubit_num)
    synthesized_circuit = Circuit(qubit_num)
    gates = UTrans(mat1).build_gate()
    synthesized_circuit.extend(gates)
    mat2 = SyntheticalUnitary.run(synthesized_circuit)
    assert np.isclose(np.linalg.det(mat1), np.linalg.det(mat2))
    assert np.allclose(mat1, mat2)
