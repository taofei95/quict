import numpy as np
import random
from scipy.stats import unitary_group

from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition
from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition

from QuICT.qcda.optimization.cnot_without_ancilla.cnot_without_ancilla import CnotWithoutAncilla


def test_unitary_decomposition_specital_circuit():
    circuit = Circuit(2)
    CX | circuit([0, 1])
    CX | circuit([1, 0])
    # CCRz(1) | circuit([1,2,3])  
    # cg_ccrz = CCRz.build_gate()
    # cg_ccrz | circuit([4,3,2])  
    # Rzz(1) | circuit([1, 2])
    # CU3(1, 0, 0) | circuit([3, 4]) 
    
    matrix1 = unitary_group.rvs(2)
    target = random.sample(range(2), 1)
    Unitary(matrix1) | circuit(target)

    UD = UnitaryDecomposition()
    gates,_ = UD.execute(matrix1)
    matrix2 = gates.matrix()
    assert np.allclose(matrix1,matrix2)
    
def test_small_matrix_run():
    rnd = 200
    for _ in range(rnd):
        mat = np.eye(2, dtype=bool)
        for _ in range(10):
            x = [0, 1]
            random.shuffle(x)
            i, j = x[0], x[1]
            mat[j, :] ^= mat[i, :]

        parallel_elimination = CnotWithoutAncilla.small_matrix_run(mat)
        for elimination_level in parallel_elimination:
            for c, t in elimination_level:
                mat[t, :] ^= mat[c, :]
    assert np.allclose(mat, np.eye(2, dtype=bool))