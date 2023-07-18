import numpy as np

from QuICT.algorithm.qft import QFT, IQFT
from QuICT.core import Circuit
from QuICT.core.gate import (
    H, X, Ry, CU3, Measure,
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
        self._circuit_cache = None
        self._circuit_params = None

    def _reconstruct(self, matrix):
        matrix_conj = matrix.T.conj()
        matrix_rec = np.kron([[0, 1], [0, 0]], matrix_conj) + np.kron([[0, 0], [1, 0]], matrix)
        return matrix_rec

    def _c_rotation(self, control, target, c: int = 1):
        """Controlled-Rotation part in HHL algorithm

        Args:
            control(int/list[int]): control qubits in multicontrol toffoli gates, in HHL it is phase qubits part
            target(int): target qubit operate CRy gate, in HHL it is ancilla qubit part
            c(int): param in |lambda|/c

        Return:
            CompositeGate
        """
        c = max(1, c)
        n = len(control)
        control_rotation_gates = CompositeGate()
        multi_control = MultiControlToffoli()(n - 1)
        for l in range(c, (1 << n) - c + 1):
            for idx in range(n):
                if ((l >> idx) & 1) == 0:
                    X | control_rotation_gates(control[idx])

            if l < (1 << n - 1):
                CU3(np.arcsin(c / l), 0, 0) | control_rotation_gates([control[0], target])
            elif l > (1 << n - 1):
                CU3(np.arcsin(c / (l - (1 << n))), 0, 0) | control_rotation_gates([control[0], target])

            multi_control | control_rotation_gates(control[1:] + [target])

            if l < (1 << n - 1):
                CU3(-np.arcsin(c / l), 0, 0) | control_rotation_gates([control[0], target])
            elif l > (1 << n - 1):
                CU3(-np.arcsin(c / (l - (1 << n))), 0, 0) | control_rotation_gates([control[0], target])

            multi_control | control_rotation_gates(control[1:] + [target])

            for idx in range(n):
                if ((l >> idx) & 1) == 0:
                    X | control_rotation_gates(control[idx])

        X | control_rotation_gates(target)
        return control_rotation_gates

    def reset(self):
        self._circuit_params = None
        self._circuit_cache = None

    def circuit(
        self,
        matrix=None,
        vector=None,
        dominant_eig=None,
        min_abs_eig=None,
        phase_qubits: int = 9,
        control_unitary=ControlledUnitaryDecomposition(recursive_basis=1),
        measure=True
    ):
        """
        Args:
            matrix(ndarray): the matrix A above, which shape must be 2^n * 2^n
            vector(array): the vector b above, which shape must be 2^n
                matrix and vector MUST have the same number of ROWS!
            dominant_eig(float/None): estimation of dominant eigenvalue
                If None, use 'np.linalg.eigvals' to obtain
            min_abs_eig(float/None): estimation of minimum absolute eigenvalue
                If None, use 'np.linalg.eigvals' to obtain
            phase_qubits(int): number of qubits representing the Phase
            control_unitary: method for preparing control-unitary gates
            measure(bool): measure ancilla qubit or not
        Returns:
            Circuit: HHL circuit
        """
        if self._circuit_cache:
            return self._circuit_cache

        self._circuit_params = (matrix, vector, dominant_eig, min_abs_eig, phase_qubits, control_unitary, measure)

        n = int(np.log2(len(matrix)))
        if (1 << n) != len(matrix) or (1 << n) != len(matrix[0]) or (1 << n) != len(vector):
            raise QuICTException(
                3999,
                f"Shape of matrix and vector should be 2^n, here are {len(matrix)}*{len(matrix[0])} and {len(vector)}"
            )

        vector /= np.linalg.norm(vector)
        if not np.allclose(matrix, matrix.T.conj(), rtol=1e-6, atol=1e-6):
            matrix = self._reconstruct(matrix)
            n += 1
            vector_ancilla = True
        else:
            vector_ancilla = False

        if not dominant_eig or not min_abs_eig:
            ev = np.abs(np.linalg.eigvalsh(matrix))
            if not dominant_eig:
                dominant_eig = np.max(ev)
            if not min_abs_eig:
                min_abs_eig = np.min(ev)

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
        scale = 1 - 1 / (1 << phase_qubits - 1)
        m = expm(matrix / dominant_eig * np.pi * 1.0j * scale)
        for idx in reversed(phase):
            U, _ = control_unitary.execute(
                np.identity(1 << n, dtype=np.complex128), m
            )
            U | unitary_matrix_gates([idx] + register)
            m = np.dot(m, m)

        # QPE
        for idx in phase:
            H | circuit(idx)
        unitary_matrix_gates | circuit
        IQFT(phase_qubits) | circuit(list(reversed(phase)))

        # Controlled-Rotation
        param_c = ((1 << phase_qubits - 1) - 1) * min_abs_eig / dominant_eig
        control_rotation = self._c_rotation(phase, ancilla, int(param_c))
        control_rotation | circuit

        # Inversed-QPE
        QFT(phase_qubits) | circuit(list(reversed(phase)))
        unitary_matrix_gates.inverse() | circuit
        for idx in phase:
            H | circuit(idx)

        if measure:
            Measure | circuit(ancilla)

        logger.info(
            f"circuit width    = {circuit.width():4}\n" +
            f"circuit size     = {circuit.size():4}\n" +
            f"hamiltonian size = {unitary_matrix_gates.size():4}\n" +
            f"CRy size         = {control_rotation.size():4}\n" +
            f"eigenvalue bits  = {phase_qubits:4}"
        )

        self._circuit_cache = circuit

        return self._circuit_cache

    def run(self):
        """
        Returns:
            list: vector x_hat, which equal to kx:
                x is the solution vector of Ax = b, and k is an unknown coefficient
        """
        if not self._circuit_cache:
            raise QuICTException(
                3999,
                "The HHL algorithm has not already generated a circuit, please run 'hhl.circuit(**args)' first."
            )

        vector, measure = self._circuit_params[1], self._circuit_params[-1]

        size = len(vector)
        circuit = self._circuit_cache
        state_vector = self.simulator.run(circuit)

        try:
            state_vector = state_vector.get()
        except:
            pass

        if measure and int(circuit[0]) == 0 or not measure:
            return np.array(state_vector[: size], dtype=np.complex128)
