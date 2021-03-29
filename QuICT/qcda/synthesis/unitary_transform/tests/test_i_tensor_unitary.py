from typing import *
# noinspection PyUnresolvedReferences
from scipy.linalg import cossin
from scipy.linalg import block_diag
from QuICT.core import *
from QuICT.algorithm import SyntheticalUnitary


def test_i_tensor_unitary():
    rnd = 20
    for _ in range(rnd):
        qubit_num = 5
        circuit1 = Circuit(qubit_num)
        circuit1.random_append(rand_size=20)
        mat1 = SyntheticalUnitary.run(circuit1)
        gates: List[BasicGate] = copy.deepcopy(circuit1.gates)
        for gate in gates:
            for idx, _ in enumerate(gate.cargs):
                gate.cargs[idx] += 1
            for idx, _ in enumerate(gate.targs):
                gate.targs[idx] += 1
        circuit2 = Circuit(qubit_num + 1)
        circuit2.extend(gates)
        mat2 = SyntheticalUnitary.run(circuit2)
        assert np.allclose(block_diag(mat1, mat1), mat2)