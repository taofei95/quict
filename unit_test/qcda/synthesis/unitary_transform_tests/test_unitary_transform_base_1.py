from typing import *
import numpy as np
from scipy.stats import unitary_group
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform
from QuICT.core import *
from QuICT.core.gate import BasicGate


def check_args(gates: Sequence[BasicGate]):
    args = set([])
    for gate in gates:
        args.update(gate.cargs)
        args.update(gate.targs)
    print(args)


def test_unitary_transform_base_1():
    rnd = 10
    for _ in range(rnd):
        qubit_num = np.random.randint(1, 7)
        mat1 = unitary_group.rvs(1 << qubit_num)
        mat1_cpy = mat1.copy()
        gates, _ = UnitaryTransform.execute(mat1, recursive_basis=1)
        mat2 = gates.matrix()
        assert np.allclose(mat1, mat1_cpy)
        # assert np.isclose(np.linalg.det(mat1), np.linalg.det(mat2))
        assert np.allclose(mat1, mat2)
