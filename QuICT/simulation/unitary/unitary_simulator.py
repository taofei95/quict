from typing import *
import numpy as np

from QuICT.core import Circuit
from QuICT.core.noise import NoiseModel
from QuICT.core.virtual_machine import VirtualQuantumMachine
from QuICT.simulation.utils import GateSimulator
from QuICT.tools.exception.core import ValueError
from QuICT.tools.exception.simulation import SampleBeforeRunError


class UnitarySimulator():
    """ Algorithms to calculate the unitary matrix of a quantum circuit, and simulate.

    Args:
        device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
        precision (str, optional): The precision for the unitary matrix, one of [single, double]. Defaults to "double".
    """
    @property
    def vector(self):
        return self._vector

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        assert device in ["CPU", "GPU"], ValueError("UnitarySimulation.device", "[CPU, GPU]", device)
        self._device = device
        assert precision in ["single", "double"], \
            ValueError("UnitarySimulation.precision", "[single, double]", precision)
        self._precision = precision
        self._gate_calculator = GateSimulator(self._device, self._precision)
        self._circuit = None
        self._vector = None
        self._quantum_machine = None

    def run(
        self,
        circuit: Union[np.ndarray, Circuit],
        state_vector: np.ndarray = None,
        quantum_machine_model: Union[NoiseModel, VirtualQuantumMachine] = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ Simulation by given unitary matrix or circuit

        Args:
            circuit (Union[np.ndarray, Circuit]): The unitary matrix or the circuit for simulation
            state_vector (ndarray): The initial state vector.
            use_previous (bool, optional): whether using previous state vector. Defaults to False.

        Returns:
            np.ndarray: The state vector after simulation
        """
        # Step 1: Deal with the Physical Machine Model
        if isinstance(circuit, Circuit) and quantum_machine_model is not None:
            noise_model = quantum_machine_model if isinstance(quantum_machine_model, NoiseModel) else \
                NoiseModel(quantum_machine_info=quantum_machine_model)
            if not noise_model.is_ideal_model():
                circuit = noise_model.transpile(circuit)
                self._quantum_machine = noise_model

        # Step 2: Generate the unitary matrix of the given circuit
        if isinstance(circuit, Circuit):
            self._circuit = circuit
            self._qubits_num = circuit.width()
            circuit.set_precision(self._precision)
            self._unitary_matrix = circuit.matrix(self._device)
        else:
            row = circuit.shape[0]
            self._qubits_num = int(np.log2(row))
            self._unitary_matrix = self._gate_calculator.normalized_matrix(circuit, self._qubits_num)

        # Step 2: Prepare the state vector
        self._original_state_vector = None
        if state_vector is not None:
            self._vector = self._gate_calculator.normalized_state_vector(state_vector, self._qubits_num)
            if self._quantum_machine is not None:
                self._original_state_vector = state_vector.copy()
        elif not use_previous or self._vector is None:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits_num)

        # Step 3: Simulation with the unitary matrix and qubit's state vector
        self._vector = self._gate_calculator.dot(
            self._unitary_matrix,
            self._vector
        )

        return self._vector

    def sample(self, shots: int = 1, target_qubits: list = None) -> list:
        """_summary_

        Args:
            shots (int): _description_

        Returns:
            _type_: _description_
        """
        assert (self._vector is not None), \
            SampleBeforeRunError("UnitarySimulation sample without run any circuit.")
        if self._quantum_machine is not None:
            return self._sample_with_noise(shots, target_qubits)

        target_qubits = target_qubits if target_qubits is not None else list(range(self._qubits_num))
        state_list = [0] * (1 << len(target_qubits))
        original_sv = self._vector.copy()
        for _ in range(shots):
            final_state = self._get_measured_result(target_qubits)
            state_list[final_state] += 1
            self._vector = original_sv.copy()

        return state_list

    def _sample_with_noise(self, shots: int, target_qubits: list) -> list:
        target_qubits = target_qubits if target_qubits is not None else list(range(self._qubits_num))
        state_list = [0] * (1 << len(target_qubits))

        for _ in range(shots):
            final_state = self._get_measured_result(target_qubits)

            # Apply readout noise
            final_state = self._quantum_machine.apply_readout_error(target_qubits, final_state)
            state_list[final_state] += 1

            # Re-generate noised circuit and initial state vector
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits_num) \
                if self._original_state_vector is None else self._original_state_vector.copy()
            noised_circuit = self._quantum_machine.transpile(self._circuit)
            self._vector = self._gate_calculator.dot(
                noised_circuit.matrix(self._device),
                self._vector
            )

        return state_list

    def _get_measured_result(self, target_qubits: list):
        final_state = 0
        for m_id in target_qubits:
            index = self._qubits_num - 1 - m_id
            measured = self._gate_calculator.apply_measure_gate(index, self._vector, self._qubits_num)
            final_state <<= 1
            final_state += measured

        return final_state
