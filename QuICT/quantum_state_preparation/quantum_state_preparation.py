import numpy as np

from QuICT.core import Circuit
from QuICT.qcda.synthesis.uniformly_gate import UniformlyRy, UniformlyRz
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform
from QuICT.quantum_state_preparation.utility import schmidt_decompose


class QuantumStatePreparation(object):
    """
    For a given quantum state |psi>, create a Circuit C such that |psi> = C |0>
    """
    @classmethod
    def with_uniformly_gates(cls, state_vector):
        """
        Quantum state preparation with uniformly gates

        Args:
            state_vector(np.ndarray): the statevector to be prapared

        Returns:
            Circuit: the preparation Circuit

        Reference:
            https://arxiv.org/abs/quant-ph/0407010v1
        """
        state_vector = np.array(state_vector)
        num_qubits = int(np.log2(state_vector.size))
        assert state_vector.ndim == 1 and 1 << num_qubits == state_vector.size,\
            ValueError('Quantum state should be an array with length 2^n')

        circuit = Circuit(num_qubits)
        # If state_vector is real, no UniformlyRz will be needed.
        if not np.allclose(state_vector.imag, 0):
            omega = np.angle(state_vector)
            for k in range(num_qubits):
                alpha = np.zeros(1 << num_qubits - k - 1)
                for j in range(1 << num_qubits - k - 1):
                    alpha[j] = (np.sum(omega[(2 * j + 1) * (1 << k):(2 * j + 1) * (1 << k) + (1 << k)] -
                                       omega[(2 * j) * (1 << k):(2 * j) * (1 << k) + (1 << k)])) / (1 << k)
                UniformlyRz.execute(alpha) | circuit
        # Now for the real state_vector
        state_vector = np.abs(state_vector)
        for k in range(num_qubits):
            alpha = np.zeros(1 << num_qubits - k - 1)
            for j in range(1 << num_qubits - k - 1):
                alpha[j] = 2 * np.arcsin(np.sqrt() / np.sqrt())

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
