import numpy as np
from scipy.stats import unitary_group

from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition


def test_unitary_decomposition_base_1():
    for qubit_num in range(2, 4):
        mat1 = unitary_group.rvs(1 << qubit_num)
        UD = UnitaryDecomposition(recursive_basis=1)
        gates, _ = UD.execute(mat1)
        mat2 = gates.matrix()
        assert np.allclose(mat1, mat2)
