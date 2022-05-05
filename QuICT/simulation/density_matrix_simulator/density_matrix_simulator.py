import numpy as np
import random

from QuICT.core.circuit.circuit import Circuit
from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.core.utils import GateType, matrix_product_to_circuit
import QuICT.ops.linalg.cpu_calculator as CPUCalculator


class DensityMatrixSimulation:

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        self._device = device
        self._precision = np.complex128 if precision == "double" else np.complex64

        if device == "CPU":
            self._computer = CPUCalculator
        else:
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator
            self._computer = GPUCalculator

    def _init_density_matrix(self, qubits):
        if self._device == "CPU":
            self._density_matrix = np.zeros((1 << qubits, 1 << qubits), dtype=self._precision)
            self._density_matrix[0, 0] = self._precision(1)
        else:
            import cupy as cp

            self._density_matrix = cp.zeros((1 << qubits, 1 << qubits), dtype=self._precision)
            self._density_matrix.put((0, 0), self._precision(1))

    def check_matrix(self, matrix):
        if(matrix.T.conjugate() != matrix):
            return False

        eigenvalues = np.linalg.eig(matrix)[0]
        for ev in eigenvalues:
            if(ev < 0):
                return False

        if(matrix.trace() != 1):
            return False

        return True

    def _measure(self, gate, qubits):
        P0 = np.array([[1, 0], [0, 0]], dtype=self._precision)

        mea_0 = matrix_product_to_circuit(P0, gate.targs, qubits)
        prob_0 = np.matmul(mea_0, self._density_matrix).trace()
        _0_1 = random.random() < prob_0
        if _0_1:
            U = np.matmul(mea_0, np.eye(1 << qubits) / np.sqrt(prob_0))
            self._density_matrix = self._computer.dot(self._computer.dot(U, self._density_matrix), U.conj().T)
        else:
            P1 = np.array([[0, 0], [0, 1]], dtype=self._precision)
            mea_1 = matrix_product_to_circuit(P1, gate.targs, qubits)
            U = np.matmul(mea_1, np.eye(1 << qubits) / np.sqrt(1 - prob_0))
            self._density_matrix = self._computer.dot(self._computer.dot(U, self._density_matrix), U.conj().T)

        return _0_1

    def run(self, circuit: Circuit, density_matrix: np.ndarray = None):
        qubits = circuit.width()
        if (density_matrix is None or not self.check_matrix(density_matrix)):
            self._init_density_matrix(qubits)
        else:
            self._density_matrix = density_matrix

        # Assume no measure gate in circuit middle, measure gate only appear last
        # circuit.gates [non-measure gates] [measure gate]
        measure_gate_list = []
        for gate in circuit.gates:
            if gate.type == GateType.measure:
                measure_gate_list.append(gate)

        circuit_matrix = UnitarySimulator().get_unitary_matrix(circuit)

        # step ops
        self._density_matrix = self._computer.dot(
            self._computer.dot(circuit_matrix, self._density_matrix),
            circuit_matrix.conj().T
        )

        # [measure gate] exist
        if measure_gate_list:
            for mea_gate in measure_gate_list:
                _0_1 = self._measure(mea_gate, circuit.width())
                circuit.qubits[gate.targ].measured = int(_0_1)

        return self._density_matrix
