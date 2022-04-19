import numpy as np

from QuICT.core import Circuit
from QuICT.quantum_state_preparation.utility import schmidt_decompose

class QuantumStatePreparation(object):
    """
    For a given quantum state |psi>, create a Circuit C such that |psi> = C |0>
    """
    @classmethod
    def with_uniformly_gates(cls, state_vector):
        """
        """

    @classmethod
    def with_unitary_transform(cls, state_vector):
        """
        """
