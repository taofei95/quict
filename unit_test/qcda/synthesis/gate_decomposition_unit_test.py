import random

import numpy as np
from scipy.stats import unitary_group

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition


def test_circuit():
    circuit = Circuit(5)
    circuit.random_append()
    CCRz(np.pi / 3) | circuit([0, 1, 2])
    CSwap | circuit([2, 3, 4])
    for _ in range(5):
        matrix = unitary_group.rvs(2 ** 3)
        target = random.sample(range(5), 3)
        Unitary(matrix) | circuit(target)

    GD = GateDecomposition()
    circuit_decomposed = GD.execute(circuit)

    assert np.allclose(circuit.matrix(), circuit_decomposed.matrix())
