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
        self._circuit = None

    def run(
        self,
        circuit: Union[np.ndarray, Circuit],
        state_vector: np.ndarray = None,
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
        # Step 1: Generate the unitary matrix of the given circuit
        if isinstance(circuit, Circuit):
            self._qubits_num = circuit.width()
            self._unitary_matrix = circuit.matrix(self._device)
        else:
            row = circuit.shape[0]
            self._qubits_num = int(np.log2(row))
            self._unitary_matrix = self._gate_calculator.normalized_matrix(circuit, self._qubits_num)

        # Step 2: Prepare the state vector
        if state_vector is not None:
            self._vector = self._gate_calculator.normalized_state_vector(state_vector, self._qubits_num)
        elif not use_previous or self._vector is None:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits_num)

        # Step 3: Simulation with the unitary matrix and qubit's state vector
        self._vector = self._gate_calculator.dot(
            self._unitary_matrix,
            self._vector
        )

        return self._vector

    def sample(self, shots: int):
        """_summary_

        Args:
            shots (int): _description_

        Returns:
            _type_: _description_
        """
        assert (self._vector is not None), \
            SampleBeforeRunError("UnitarySimulation sample without run any circuit.")

        original_sv = self._vector.copy()
        counts = [0] * (1 << self._qubits_num)
        for _ in range(shots):
            measured_result = 0
            for i in range(self._qubits_num - 1, -1, -1):
                measured_result <<= 1
                measured_result += self._gate_calculator.apply_measure_gate(i, self._vector, self._qubits_num)

            counts[measured_result] += 1
            self._vector = original_sv.copy()

        return counts
