from typing import *
import numpy as np

from QuICT.core import Circuit, circuit
import QuICT.ops.linalg.cpu_calculator as CPUCalculator


class UnitarySimulator():
    """ Algorithms to calculate the unitary matrix of a quantum circuit, and simulate.

    Args:
        device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
        precision (str, optional): The precision for the unitary matrix, one of [single, double]. Defaults to "double".
    """

    def __init__(
            self,
            device: str = "CPU",
            precision: str = "double"
    ):
        self._device = device
        self._precision = np.complex128 if precision == "double" else np.complex64
        self._vector = None
        self._circuit = None

        if device == "CPU":
            self._computer = CPUCalculator
            self._array_helper = np
        else:
            import cupy as cp
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator

            self._computer = GPUCalculator
            self._array_helper = cp

    def initial_vector_state(self):
        """ Initial the state vector for simulation through UnitarySimulator,
        must after initial_circuit
        """
        self._vector = self._array_helper.zeros(1 << self._qubits_num, dtype=self._precision)
        if self._device == "CPU":
            self._vector[0] = self._precision(1)
        else:
            self._vector.put(0, self._precision(1))

    def run(self, matrix: Union[np.ndarray, Circuit], use_previous: bool = False) -> np.ndarray:
        """ Simulation by given unitary matrix or circuit

        Args:
            matrix (Union[np.ndarray, Circuit]): The unitary matrix or the circuit for simulation
            use_previous (bool, optional): whether using previous state vector. Defaults to False.

        Returns:
            np.ndarray: The state vector after simulation
        """
        # Step 1: Generate the unitary matrix of the given circuit
        if isinstance(matrix, Circuit):
            self._qubits_num = matrix.width()
            self._unitary_matrix = matrix.matrix(self._device)
            assert 2 ** self._qubits_num == self._unitary_matrix.shape[0]
        else:
            row, col = matrix.shape
            self._qubits_num = int(np.log2(row))
            assert row == col and 2 ** self._qubits_num == col

            self._unitary_matrix = self._array_helper.array(matrix)

        if not use_previous or self._vector is None:
            self.initial_vector_state()

        if self._unitary_matrix is None:
            return self._vector

        # Step 2: Simulation with the unitary matrix and qubit's state vector
        self._vector = self._computer.dot(
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
        assert self._circuit is not None
        original_sv = self._vector.copy()
        counts = [0] * (1 << self._qubits_num)
        for _ in range(shots):
            for i in range(self._qubits_num):
                self._measure(i)

            counts[int(self._circuit.qubits)] += 1
            self._vector = original_sv.copy()

        return counts

    def _measure(self, index):
        if self._device == "CPU":
            result = self._computer.measure_gate_apply(
                index,
                self._vector
            )
        else:
            from QuICT.ops.gate_kernel import apply_measuregate, measured_prob_calculate
            prob = measured_prob_calculate(
                index,
                self._vector,
                self._qubits_num
            )
            result = apply_measuregate(
                index,
                self._vector,
                self._qubits_num,
                prob=prob
            )

        self._circuit.qubits[self._qubits_num - 1 - index].measured = int(result)
