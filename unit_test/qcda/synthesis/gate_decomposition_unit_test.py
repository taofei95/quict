
import pytest
import random

import numpy as np
from scipy.stats import unitary_group

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition


def test_unitary():
    matrix = unitary_group.rvs(2 ** 5)
    gates_decomposed = GateDecomposition.execute(matrix)

    assert np.allclose(matrix, gates_decomposed.matrix())


def test_circuit():
    circuit = Circuit(5)
    circuit.random_append()
    CCRz(np.pi / 3) | circuit([0, 1, 2])
    CSwap | circuit([2, 3, 4])
    for _ in range(5):
        matrix = unitary_group.rvs(2 ** 3)
        target = random.sample(range(5), 3)
        Unitary(matrix) | circuit(target)

    gates_decomposed = GateDecomposition.execute(circuit)
    circuit_decomposed = Circuit(5)
    gates_decomposed | circuit_decomposed

    assert np.allclose(circuit.matrix(), circuit_decomposed.matrix())


if __name__ == "__main__":
    pytest.main(["./unit_test.py"])
