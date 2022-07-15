import numpy as np
import random

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import BasicGate, UnitaryGate, Unitary
from QuICT.core.noise import NoiseModel
from QuICT.core.operator import NoiseGate
from QuICT.simulation.unitary import UnitarySimulator
from QuICT.core.utils import GateType, matrix_product_to_circuit
import QuICT.ops.linalg.cpu_calculator as CPUCalculator


class DensityMatrixSimulation:
    """ The Density Matrix Simulator

    Args:
        device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
        precision (str, optional): The precision for the density matrix, one of [single, double]. Defaults to "double".
        accumulated_mode (bool): If True, calculated density matrix with Kraus Operators in NoiseGate.
            if True, p = \\sum Ki p Ki^T.conj(). Default to be False.
    """
    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double",
        accumulated_mode: bool = False
    ):
        self._device = device
        self._precision = np.complex128 if precision == "double" else np.complex64
        self._accumulated_mode = accumulated_mode
        self._density_matrix = None

        if device == "CPU":
            self._computer = CPUCalculator
            self._array_helper = np
        else:
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator
            import cupy as cp

            self._computer = GPUCalculator
            self._array_helper = cp

    def initial_circuit(self, circuit: Circuit, noise_model: NoiseModel):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._circuit = circuit if noise_model is None else noise_model.transpile(circuit)
        self._qubits = int(circuit.width())
        self._last_call_per_qubit = [None] * self._qubits

    def initial_density_matrix(self, qubits: int):
        """ Initial density matrix by given qubits number.

        Args:
            qubits (int): the number of qubits.
        """
        self._density_matrix = self._array_helper.zeros((1 << qubits, 1 << qubits), dtype=self._precision)
        if self._device == "CPU":
            self._density_matrix[0, 0] = self._precision(1)
        else:
            self._density_matrix.put((0, 0), self._precision(1))

    def check_matrix(self, matrix):
        """ Density Matrix Validation. """
        if(matrix.T.conjugate() != matrix):
            return False

        eigenvalues = np.linalg.eig(matrix)[0]
        for ev in eigenvalues:
            if(ev < 0):
                return False

        if(matrix.trace() != 1):
            return False

        return True

    def run(
        self,
        circuit: Circuit,
        noise_model: NoiseModel = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ Simulating the given circuit through density matrix simulator.

        Args:
            circuit (Circuit): The quantum circuit.
            noise_model (NoiseModel, optional): The NoiseModel contains NoiseErrors. Defaults to None.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Returns:
            np.ndarray: the density matrix after simulating
        """
        self.initial_circuit(circuit, noise_model)
        # Initial density matrix
        if (self._density_matrix is None or not use_previous):
            self.initial_density_matrix(self._qubits)
        else:
            assert self.check_matrix(self._density_matrix)

        # Start simulator
        based_circuit = Circuit(self._qubits)
        for gate in self._circuit.gates:
            # Store continuous BasicGates into based_circuit
            if isinstance(gate, BasicGate) and gate.type != GateType.measure:
                gate | based_circuit
                continue

            if not self._accumulated_mode and isinstance(gate, NoiseGate):
                ugate = self.apply_noise_without_accumulated(gate)
                ugate | based_circuit
                gate.gate | based_circuit
                continue

            if based_circuit.size() > 0:
                self.apply_gates(based_circuit)
                based_circuit = Circuit(self._qubits)

            if gate.type == GateType.measure:
                self.apply_measure(gate, self._qubits)
                self._last_call_per_qubit[gate.targ] = GateType.measure
            elif isinstance(gate, NoiseGate):
                self.apply_noise(gate, self._qubits)
            else:
                raise KeyError("Unsupportted operator in Density Matrix Simulator.")

        if based_circuit.size() > 0:
            self.apply_gates(based_circuit)

        # Check Readout Error in the NoiseModel
        if noise_model is not None:
            noise_model.apply_readout_error(circuit.qubits)

        return self._density_matrix

    def apply_gates(self, circuit: Circuit):
        """ Simulating Circuit with BasicGates

        dm = M*dm(M.conj)^T

        Args:
            circuit (Circuit): The circuit only have BasicGate.
        """
        circuit_matrix = UnitarySimulator().get_unitary_matrix(circuit)

        self._density_matrix = self._computer.dot(
            self._computer.dot(circuit_matrix, self._density_matrix),
            circuit_matrix.conj().T
        )

    def apply_noise(self, noise_gate: NoiseGate, qubits: int):
        """ Simulating NoiseGate.

        dm = /sum K*dm*(K.conj)^T

        Args:
            noise_gate (NoiseGate): The NoiseGate
            qubits (int): The number of qubits in the circuit.
        """
        gate_args = noise_gate.cargs + noise_gate.targs
        noised_matrix = self._array_helper.zeros_like(self._density_matrix)
        for kraus_matrix in noise_gate.noise_matrix:
            umat = matrix_product_to_circuit(kraus_matrix, gate_args, qubits)

            noised_matrix += self._computer.dot(
                self._computer.dot(umat, self._density_matrix),
                umat.conj().T
            )

        self._density_matrix = noised_matrix.copy()

    def apply_noise_without_accumulated(self, gate: NoiseGate) -> UnitaryGate:
        prob = np.random.random()
        error_matrix = gate.prob_mapping_operator(prob)
        gate_args = gate.targs
        return Unitary(error_matrix) & gate_args

    def apply_measure(self, gate, qubit: int):
        """ Simulating the MeasureGate.

        Args:
            gate (BasicGate): The MeasureGate.
            qubits (int): The index of qubits in the circuit.

        Returns:
            int: The measured result.
        """
        P0 = np.array([[1, 0], [0, 0]], dtype=self._precision)

        mea_0 = matrix_product_to_circuit(P0, gate.targs, qubit)
        prob_0 = np.matmul(mea_0, self._density_matrix).trace()
        _0_1 = random.random() < prob_0
        if _0_1:
            U = np.matmul(mea_0, np.eye(1 << qubit) / np.sqrt(prob_0))
            self._density_matrix = self._computer.dot(self._computer.dot(U, self._density_matrix), U.conj().T)
        else:
            P1 = np.array([[0, 0], [0, 1]], dtype=self._precision)
            mea_1 = matrix_product_to_circuit(P1, gate.targs, qubit)
            U = np.matmul(mea_1, np.eye(1 << qubit) / np.sqrt(1 - prob_0))
            self._density_matrix = self._computer.dot(self._computer.dot(U, self._density_matrix), U.conj().T)

        self._circuit.qubits[qubit].measured = int(_0_1)

    def sample(self, shots: int) -> list:
        assert (self._density_matrix is not None)
        original_dm = self._density_matrix.copy()
        state_list = [0] * self._density_matrix.shape[0]
        measured_idx = [
            i for i in range(self._qubits)
            if self._last_call_per_qubit[i] == GateType.measure
        ]

        for _ in range(shots):
            for m_id in measured_idx:
                index = self._qubits - 1 - m_id
                prob = self.get_measured_prob(index).get()
                _ = self.apply_specialgate(index, GateType.measure, prob)

            state_list[int(self._circuit.qubits)] += 1
            self._density_matrix = original_dm.copy()

        return state_list

