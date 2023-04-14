import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import (
    H, X, Ry, CU3, Measure,
    QFT, IQFT,
    CompositeGate, MultiControlToffoli)
from QuICT.qcda.synthesis.quantum_state_preparation import QuantumStatePreparation
from QuICT.qcda.synthesis.unitary_decomposition.controlled_unitary import ControlledUnitaryDecomposition
from QuICT.tools import Logger
from QuICT.tools.exception import QuICTException

from scipy.linalg import expm


logger = Logger('hhl')


class HHL:
    """ original HHL algorithm

    References:
        [1] Quantum Algorithm for Linear Systems of Equations: https://doi.org/10.1103/PhysRevLett.103.150502
        [2] Quantum circuit design for solving linear systems of equations: https://doi.org/10.1080/00268976.2012.668289
    """
    def __init__(self, simulator=None) -> None:
        self.simulator = simulator

    def _reconstruct(self, matrix):
        matrix_conj = matrix.T.conj()
        matrix_rec = np.kron([[0, 1], [0, 0]], matrix_conj) + np.kron([[0, 0], [1, 0]], matrix)
        return matrix_rec

    def _c_rotation(self, control, target):
        """Controlled-Rotation part in HHL algorithm

        Args:
            control(int/list[int]): control qubits in multicontrol toffoli gates, in HHL it is phase qubits part
            target(int): target qubit operate CRy gate, in HHL it is ancilla qubit part

        Return:
            CompositeGate
        """
        c = 1
        n = len(control)
        control_rotation_gates = CompositeGate()
        multi_control = MultiControlToffoli()(n - 1)
        for l in range(c, (1 << n) - c + 1):
            for idx in range(n):
                if ((l >> idx) & 1) == 0:
                    X & [control[idx]] | control_rotation_gates

            if l < (1 << (n - 1)):
                CU3(np.arcsin(c / l), 0, 0) & [control[0], target] | control_rotation_gates
            else:
                CU3(np.arcsin(c / (l - (1 << n))), 0, 0) & [control[0], target] | control_rotation_gates

            multi_control | control_rotation_gates(control[1:] + [target])

            if l < (1 << (n - 1)):
                CU3(-np.arcsin(c / l), 0, 0) & [control[0], target] | control_rotation_gates
            else:
                CU3(-np.arcsin(c / (l - (1 << n))), 0, 0) & [control[0], target] | control_rotation_gates

            multi_control | control_rotation_gates(control[1:] + [target])

            for idx in range(n):
                if ((l >> idx) & 1) == 0:
                    X & control[idx] | control_rotation_gates

        X & target | control_rotation_gates
        return control_rotation_gates

    def circuit(
        self,
        matrix,
        vector,
        dominant_eig=None,
        phase_qubits: int = 9,
        measure=True
    ):
        """
        Args:
            matrix(ndarray): the matrix A above, which shape must be 2^n * 2^n
            vector(array): the vector b above, which shape must be 2^n
                matrix and vector MUST have the same number of ROWS!
            dominant_eig(float/None): estimation of dominant eigenvalue
                If None, use 'np.linalg.eigvals' to obtain
            phase_qubits(int): number of qubits representing the Phase
            measure(bool): measure ancilla qubit or not
        Returns:
            Circuit: HHL circuit
        """
        n = int(np.log2(len(matrix)))
        if (1 << n) != len(matrix) or (1 << n) != len(matrix[0]) or (1 << n) != len(vector):
            raise QuICTException(
                f"shape of matrix and vector should be 2^n, here are {len(matrix)}*{len(matrix[0])} and {len(vector)}")

        vector /= np.linalg.norm(vector)
        if not np.allclose(matrix, matrix.T.conj()):
            matrix = self._reconstruct(matrix)
            n += 1
            vector_ancilla = True
        else:
            vector_ancilla = False

        if not dominant_eig:
            dominant_eig = np.max(np.abs(np.linalg.eigvalsh(matrix)))

        circuit = Circuit(1 + phase_qubits + n)
        ancilla = 0
        phase = list(range(1, 1 + phase_qubits))
        register = list(range(1 + phase_qubits, 1 + phase_qubits + n))

        # State preparation
        if vector_ancilla:
            X | circuit(register[0])
        if len(vector) > 2:
            QuantumStatePreparation().execute(vector) | circuit(register[vector_ancilla:])
        else:
            Ry(2 * np.arcsin(vector[1])) | circuit(register[-1])

        # prepare Controlled-Unitary Gate
        unitary_matrix_gates = CompositeGate()
        m = expm(matrix / dominant_eig * 0.5j)
        for idx in reversed(phase):
            U, _ = ControlledUnitaryDecomposition().execute(
                np.identity(1 << n, dtype=np.complex128), m
            )
            U | unitary_matrix_gates([idx] + register)
            m = np.dot(m, m)

        # QPE
        for idx in phase:
            H | circuit(idx)
        unitary_matrix_gates | circuit
        IQFT.build_gate(len(phase)) | circuit(list(reversed(phase)))

        # Controlled-Rotation
        control_rotation = self._c_rotation(phase, ancilla)
        control_rotation | circuit

        # Inversed-QPE
        QFT.build_gate(len(phase)) | circuit(list(reversed(phase)))
        unitary_matrix_gates.inverse() | circuit
        for idx in phase:
            H | circuit(idx)

        if measure:
            Measure | circuit(ancilla)

        logger.info(
            f"circuit width    = {circuit.width():4}\n" +
            f"circuit size     = {circuit.size():4}\n" +
            f"hamiltonian size = {unitary_matrix_gates.size():4}\n" +
            f"CRy size         = {control_rotation.size():4}"
        )

        return circuit

    def run(
        self,
        matrix,
        vector,
        dominant_eig=None,
        phase_qubits: int = 9,
        measure=True
    ):
        """ hhl algorithm to solve linear equation such as Ax=b,
            where A is the given matrix and b is the given vector
        Args:
            matrix(ndarray): the matrix A above, which shape must be 2^n * 2^n
            vector(array): the vector b above, which shape must be 2^n
                matrix and vector MUST have the same number of ROWS!
            dominant_eig(float/None): estimation of dominant eigenvalue
                If None, use 'np.linalg.eigvals' to obtain
            phase_qubits(int): number of qubits representing the Phase
            measure(bool): measure ancilla qubit or not
        Returns:
            list: vector x_hat, which equal to kx:
                x is the solution vector of Ax = b, and k is an unknown coefficient
        """
        size = len(vector)

        circuit = self.circuit(matrix, vector, dominant_eig, phase_qubits, measure)

        state_vector = self.simulator.run(circuit)

        if self.simulator._device is "GPU":
            state_vector = state_vector.get()

        if measure and int(circuit[0]) == 0 or not measure:
            return np.array(state_vector[: size], dtype=np.complex128)
