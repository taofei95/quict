import numpy as np
from scipy.stats import unitary_group

from QuICT.quantum_state_preparation import QuantumStatePreparation


def test_with_uniformly_gates():
    for n in range(3, 6):
        for _ in range(100):
            state_vector = unitary_group.rvs(1 << n)[0]
