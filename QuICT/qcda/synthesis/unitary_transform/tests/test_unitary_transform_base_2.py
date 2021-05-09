from scipy.stats import unitary_group
from ..unitary_transform import UTrans
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_unitary_transform_base_2():
    rnd = 50
    for _ in range(rnd):
        qubit_num = np.random.randint(1, 7)
        mat1 = unitary_group.rvs(1 << qubit_num)
        gates, _ = UTrans(mat1, recursive_basis=2)
        mat2 = gates.matrix()
        assert np.allclose(mat1, mat2)
