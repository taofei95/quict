import numpy as np

from QuICT.core.gate import CompositeGate, Phase
from QuICT.qcda.synthesis.uniformly_gate import UniformlyRy, UniformlyRz
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform
from QuICT.quantum_state_preparation.utility import schmidt_decompose


class QuantumStatePreparation(object):
    """
    For a given quantum state |psi>, create a CompositeGate C such that |psi> = C |0>
    """
    @classmethod
    def with_uniformly_gates(cls, state_vector):
        """
        Quantum state preparation with uniformly gates

        Args:
            state_vector(np.ndarray): the statevector to be prapared

        Returns:
            CompositeGate: the preparation CompositeGate

        Reference:
            https://arxiv.org/abs/quant-ph/0407010v1
        """
        state_vector = np.array(state_vector)
        num_qubits = int(np.log2(state_vector.size))
        assert state_vector.ndim == 1 and 1 << num_qubits == state_vector.size,\
            ValueError('Quantum state should be an array with length 2^n')

        gates = CompositeGate()
        omega = np.angle(state_vector)
        state_vector = np.abs(state_vector)
        # Now for the non-negative real state_vector
        denominator = np.linalg.norm(state_vector)
        for k in range(num_qubits - 1, -1, -1):
            numerator = np.linalg.norm(state_vector.reshape(1 << num_qubits - k, 1 << k), axis=1)
            alpha = np.where(denominator != 0, 2 * np.arcsin(numerator[1::2] / denominator), 0)
            gates.extend(UniformlyRy.execute(alpha))
            denominator = numerator
        # If state_vector is real and non-negative, no UniformlyRz will be needed.
        if not np.allclose(omega, 0):
            for k in range(num_qubits):
                alpha = np.sum(omega.reshape(1 << num_qubits - k, 1 << k), axis=1)
                alpha = (alpha[1::2] - alpha[0::2]) / (1 << k)
                gates.extend(UniformlyRz.execute(alpha))
            gates.append(Phase(np.average(omega)) & 0)

        return gates

    @classmethod
    def with_unitary_decomposition(cls, state_vector):
        """
        Quantum state preparation with unitary decomposition

        Args:
            state_vector(np.ndarray): the statevector to be prapared

        Returns:
            Circuit: the preparation Circuit

        Reference:
            https://arxiv.org/abs/1003.5760
        """
        state_vector = np.array(state_vector)
        num_qubits = int(np.log2(state_vector.size))
        assert state_vector.ndim == 1 and 1 << num_qubits == state_vector.size,\
            ValueError('Quantum state should be an array with length 2^n')
