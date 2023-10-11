from typing import *
import numpy as np

from QuICT.core import Circuit
from QuICT.simulation.utils import GateSimulator
from QuICT.tools.exception.core import ValueError
from QuICT.tools.exception.simulation import SampleBeforeRunError


class UnitarySimulator():
    """ Algorithms to calculate the unitary matrix of a quantum circuit, and simulate. """
    @property
    def vector(self):
        return self._vector

    @property
    def device(self):
        return self._gate_calculator.device

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        """
        Args:
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
            precision (str, optional): The precision for the unitary matrix. Defaults to "double".
        """
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
            circuit.precision = self._precision
            self._unitary_matrix = circuit.matrix(self._device)
        else:
            row = circuit.shape[0]
            self._qubits_num = int(np.log2(row))
            self._unitary_matrix = self._gate_calculator.normalized_matrix(circuit, self._qubits_num)

        # Step 2: Prepare the state vector
        self._original_state_vector = None
        if quantum_state is not None:
            self._vector = self._gate_calculator.normalized_state_vector(quantum_state.copy(), self._qubits_num)
        elif not use_previous:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits_num)

        # Step 3: Simulation with the unitary matrix and qubit's state vector
        self._vector = self._gate_calculator.dot(
            self._unitary_matrix,
            self._vector
        )

        return self._vector

    def sample(self, shots: int = 1) -> list:
        """ Sample the measured result from current state vector, please first run simulator.run().

        Args:
            shots (int): The sample times for current state vector.

        Returns:
            List[int]: The measured result list with length equal to 2 ** self._qubits
        """
        assert (self._vector is not None), \
            SampleBeforeRunError("UnitarySimulation sample without run any circuit/matrix.")
        state_list = [0] * (1 << self._qubits_num)
        sample_result = self._gate_calculator.sample_for_statevector(shots, self._qubits_num, self._vector)
        for res in sample_result:
            state_list[res] += 1

        return state_list
