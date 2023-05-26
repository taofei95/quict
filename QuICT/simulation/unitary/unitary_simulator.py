from typing import *
import numpy as np

from QuICT.core import Circuit
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
        self._vector = None

    def run(
        self,
        circuit: Union[np.ndarray, Circuit],
        quantum_state: np.ndarray = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ Simulation by given unitary matrix or circuit

        Args:
            circuit (Union[np.ndarray, Circuit]): The unitary matrix or the circuit for simulation
            quantum_state (ndarray): The initial quantum state vector.
            use_previous (bool, optional): whether using previous state vector. Defaults to False.

        Returns:
            np.ndarray: The state vector after simulation
        """
        # Step 1: Generate the unitary matrix of the given circuit
        if isinstance(circuit, Circuit):
            self._qubits_num = circuit.width()
            circuit.set_precision(self._precision)
            self._unitary_matrix = circuit.matrix(self._device)
        else:
            row = circuit.shape[0]
            self._qubits_num = int(np.log2(row))
            self._unitary_matrix = self._gate_calculator.normalized_matrix(circuit, self._qubits_num)

        # Step 2: Prepare the state vector
        self._original_state_vector = None
        if quantum_state is not None:
            self._vector = self._gate_calculator.normalized_state_vector(quantum_state, self._qubits_num)
        elif not use_previous:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits_num)

        # Step 3: Simulation with the unitary matrix and qubit's state vector
        self._vector = self._gate_calculator.dot(
            self._unitary_matrix,
            self._vector
        )

        return self._vector

    def sample(self, shots: int = 1, target_qubits: list = None) -> list:
        """Sample the measured result from current state vector, please first run simulator.run().

        **WARNING**: Please make sure the target qubits are not been measured before simulator.sample().

        Args:
            shots (int): The sample times for current state vector.
            target_qubits (List[int]): The indexes of qubits which want to be measured. If it is None, there
            will measured all qubits in previous circuits.

        Returns:
            List[int]: The measured result list with length equal to 2 ** len(target_qubits)
        """
        assert (self._vector is not None), \
            SampleBeforeRunError("UnitarySimulation sample without run any circuit/matrix.")
        target_qubits = target_qubits if target_qubits is not None else list(range(self._qubits_num))
        state_list = [0] * (1 << len(target_qubits))
        original_sv = self._vector.copy()
        for _ in range(shots):
            final_state = self._get_measured_result(target_qubits)
            state_list[final_state] += 1
            self._vector = original_sv.copy()

        return state_list

    def _get_measured_result(self, target_qubits: list):
        final_state = 0
        for m_id in target_qubits:
            index = self._qubits_num - 1 - m_id
            measured = self._gate_calculator.apply_measure_gate(index, self._vector, self._qubits_num)
            final_state <<= 1
            final_state += measured

        return final_state
