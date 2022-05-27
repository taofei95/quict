import numpy as np
import random

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.noise import NoiseModel
from QuICT.core.operator import NoiseGate
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

    def run(self, circuit: Circuit, noise_model: NoiseModel = None, density_matrix: np.ndarray = None):
        qubits = circuit.width()
        # Initial density matrix
        if (density_matrix is None or not self.check_matrix(density_matrix)):
            self._init_density_matrix(qubits)
        else:
            self._density_matrix = density_matrix

        # apply noise model
        if noise_model is not None:
            circuit = noise_model.transpile(circuit)

        based_circuit = Circuit(qubits)
        for gate in circuit.gates:
            if gate.type == GateType.measure:
                measured_state = self.apply_measure(gate, qubits)
                circuit.qubits[gate.targ].measured = int(measured_state)
                continue

            if isinstance(gate, BasicGate):
                gate | based_circuit
            elif isinstance(gate, NoiseGate):
                if based_circuit.size() > 0:
                    self.apply_gates(based_circuit)
                    based_circuit = Circuit(qubits)

                self.apply_noise(gate, qubits)
            else:
                raise KeyError("Unsupportted operator in Density Matrix Simulator.")

        if based_circuit.size() > 0:
            self.apply_gates(based_circuit)
            based_circuit = Circuit(qubits)

        return self._density_matrix

    def apply_gates(self, circuit: Circuit):
        circuit_matrix = UnitarySimulator().get_unitary_matrix(circuit)

        # step ops
        self._density_matrix = self._computer.dot(
            self._computer.dot(circuit_matrix, self._density_matrix),
            circuit_matrix.conj().T
        )

    def apply_noise(self, noise_gate: NoiseGate, qubits: int):
        gate_args = noise_gate.targs
        for kraus_matrix in noise_gate.noise_matrix:
            umat = matrix_product_to_circuit(kraus_matrix, gate_args, qubits)
            self._density_matrix += self._computer.dot(
                self._computer.dot(umat, self._density_matrix),
                umat.conj().T
            )

    def apply_measure(self, gate, qubits):
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
