import numpy as np
from scipy.stats import unitary_group

from QuICT.quantum_state_preparation.utility import schmidt_decompose


def test_schmidt_decompose():
    for n in range(3, 6):
        for _ in range(100):
            A_qubits = 2
            state_vector = unitary_group.rvs(1 << n)[0]
            l, iA, iB = schmidt_decompose(state_vector, A_qubits)
            res = np.zeros(1 << n, dtype=np.complex128)
            for i in range(len(l)):
                res += l[i] * np.kron(iA[i], iB[i])
            assert np.allclose(state_vector, res)
