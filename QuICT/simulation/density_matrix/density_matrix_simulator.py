from typing import Union
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.noise import NoiseModel
from QuICT.core.operator import NoiseGate
from QuICT.core.virtual_machine import VirtualQuantumMachine
from QuICT.core.utils import GateType, matrix_product_to_circuit
from QuICT.simulation.utils import GateSimulator
from QuICT.tools.exception.core import TypeError, ValueError
from QuICT.tools.exception.simulation import SampleBeforeRunError


class DensityMatrixSimulator:
    """ The Density Matrix Simulator """
    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double",
        accumulated_mode: bool = False
    ):
        """
        Args:
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
            precision (str, optional): The precision for the density matrix, one of [single, double]. Defaults to "double".
            accumulated_mode (bool): If True, calculated density matrix with Kraus Operators in NoiseGate.
                if True, p = \\sum Ki p Ki^T.conj(). Default to be False.
                [Important: set accumulated_mode for True if you need sample result, otherwise, using False for
                fast simulate time.]
        """
        self._gate_calculator = GateSimulator(device, precision)
        self._accumulated_mode = accumulated_mode
        self._density_matrix = None
        self._quantum_machine = None

    def initial_circuit(self, circuit: Circuit):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._origin_circuit = circuit
        self._circuit = circuit if self._quantum_machine is None else self._quantum_machine.transpile(circuit)
        self._qubits = circuit.width()

    def run(
        self,
        circuit: Circuit,
        quantum_state: np.ndarray = None,
        quantum_machine_model: Union[NoiseModel, VirtualQuantumMachine] = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ Simulating the given circuit through density matrix simulator.

        Args:
            circuit (Circuit): The quantum circuit.
            density_matrix (np.ndarray): The initial density matrix.
            quantum_machine_model (NoiseModel, optional): The NoiseModel contains NoiseErrors. Defaults to None.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Returns:
            np.ndarray: the density matrix after simulating
        """
        # Deal with the Physical Machine Model
        if quantum_machine_model is not None:
            noise_model = quantum_machine_model if isinstance(quantum_machine_model, NoiseModel) else \
                NoiseModel(quantum_machine_info=quantum_machine_model)
            if not noise_model.is_ideal_model():
                self._quantum_machine = noise_model

        # Initial Quantum Circuit
        self.initial_circuit(circuit)

        # Initial Density Matrix
        if quantum_state is not None:
            self._gate_calculator.validate_density_matrix(quantum_state)
            self._density_matrix = self._gate_calculator.normalized_matrix(quantum_state, self._qubits)
        elif (self._density_matrix is None or not use_previous):
            self._density_matrix = self._gate_calculator.get_allzero_density_matrix(self._qubits)

        self._run(self._circuit)

        return self._density_matrix

    def _run(self, noised_circuit):
        # Start simulator
        circuit = Circuit(self._qubits)
        for gate in noised_circuit.gates:
            # Store continuous BasicGates into cgate
            if isinstance(gate, BasicGate) and gate.type != GateType.measure:
                gate | circuit
                continue

            if circuit.size() > 0:
                self.apply_gates(circuit)
                circuit._gates = []

            if gate.type == GateType.measure:
                self.apply_measure(gate.targ)
            elif isinstance(gate, NoiseGate):
                self.apply_noise(gate)
            else:
                raise TypeError("DensityMatrixSimulator.run.circuit", "[BasicGate, NoiseGate]", type(gate))

        if circuit.size() > 0:
            self.apply_gates(circuit)

    def apply_gates(self, circuit: Circuit):
        """ Simulating Circuit with BasicGates

        dm = M*dm(M.conj)^T

        Args:
            cgate (CompositeGate): The CompositeGate.
        """
        cir_matrix = circuit.matrix(self._gate_calculator.device)
        self._density_matrix = self._gate_calculator.dot(
            self._gate_calculator.dot(cir_matrix, self._density_matrix),
            cir_matrix.conj().T
        )

    def apply_noise(self, noise_gate: NoiseGate):
        """ Simulating NoiseGate.

        dm = /sum K*dm*(K.conj)^T

        Args:
            noise_gate (NoiseGate): The NoiseGate
            qubits (int): The number of qubits in the circuit.
        """
        gate_args = noise_gate.targs
        noised_matrix = self._gate_calculator.get_empty_density_matrix(self._qubits)
        for kraus_matrix in noise_gate.noise_matrix:
            umat = matrix_product_to_circuit(kraus_matrix, gate_args, self._qubits, self._device)

            noised_matrix += self._gate_calculator.dot(
                self._gate_calculator.dot(umat, self._density_matrix),
                umat.conj().T
            )

        self._density_matrix = noised_matrix.copy()

    def apply_measure(self, index: int):
        """ Simulating the MeasureGate.

        Args:
            index (int): The index of measured qubit.
        """
        _1, self._density_matrix = self._gate_calculator.apply_measure_gate_for_dm(
            index, self._density_matrix, self._qubits
        )
        if self._quantum_machine is not None:
            _1 = self._quantum_machine.apply_readout_error(index, int(_1))

        self._origin_circuit.qubits[index].measured = _1

    def sample(self, shots: int, target_qubits: list = None) -> list:
        assert (self._density_matrix is not None), \
            SampleBeforeRunError("DensityMatrixSimulator sample without run any circuit.")
        if self._accumulated_mode or self._quantum_machine is None:
            original_dm = self._density_matrix.copy()

        target_qubits = target_qubits if target_qubits is not None else list(range(self._qubits))
        state_list = [0] * (1 << len(target_qubits))
        for _ in range(shots):
            final_state = 0
            for m_id in target_qubits:
                measured, self._density_matrix = self._gate_calculator.apply_measure_gate_for_dm(
                    m_id, self._density_matrix, self._qubits
                )
                final_state <<= 1
                final_state += int(measured)

            if self._quantum_machine is not None:
                final_state = self._quantum_machine.apply_readout_error(target_qubits, final_state)

            state_list[final_state] += 1
            if self._accumulated_mode or self._quantum_machine is None:
                self._density_matrix = original_dm.copy()
            else:
                self._density_matrix = self._gate_calculator.get_allzero_density_matrix(self._qubits)
                noised_circuit = self._quantum_machine.transpile(self._origin_circuit, self._accumulated_mode) \
                    if self._quantum_machine is not None else self._circuit

                self._run(noised_circuit)

        return state_list
