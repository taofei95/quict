import numpy as np
from typing import Union

from QuICT.ops.utils import LinAlgLoader


class QubitTensor:
    @property
    def tensor_data(self) -> np.ndarray:
        return self._tensor_data

    @tensor_data.setter
    def tensor_data(self, data):
        self._tensor_data = data

    @property
    def ldim(self) -> int:
        return self._tensor_data.shape[0]

    @property
    def dim(self) -> int:
        return self._tensor_data.shape[1]

    @property
    def rdim(self) -> int:
        return self._tensor_data.shape[2]

    def __init__(self, state: np.ndarray):
        self._tensor_data = state


class Normalize:
    @property
    def matrix_data(self) -> np.ndarray:
        return self._matrix_data

    @matrix_data.setter
    def matrix_data(self, data):
        self._matrix_data = data

    @property
    def ldim(self) -> int:
        return self._matrix_data.shape[0]

    @property
    def rdim(self) -> int:
        return self._matrix_data.shape[0]

    def __init__(self, norm_coeff: np.ndarray):
        self._matrix_data = norm_coeff


class MPSSiteStructure:
    __SWAP_MATRIX = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)

    @property
    def qubits(self) -> int:
        return self._qubits

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        """ Initial the MPSSiteStructure Class

        Args:
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
            precision (str, optional): The precision type, one of [single, double]. Defaults to "double".
        """
        assert device in ["CPU", "GPU"]
        self._device = device
        assert precision in ["double", "single"]
        self._precision = precision
        self._dtype = np.complex128 if self._precision == "double" else np.complex64

        self._linear_algorithm = LinAlgLoader(device=device)
        if self._device == "GPU":
            import cupy as cp

            self._array_helper = cp
        else:
            self._array_helper = np

    def initial_mps(self, qubits: int, quantum_state: Union[str, np.ndarray] = None):
        """ Generate the initial state for MPS by given quantum state or default state.

        Args:
            qubits (int): The number of qubits.
            quantum_state (Union[str, np.ndarray], optional): The initial quantum state. Defaults to None.
        """
        self._qubits = qubits
        if quantum_state is not None:
            # TODO: add special quantum state like GHZ or ... later
            if not isinstance(quantum_state, str):
                quantum_state = self._array_helper.array(quantum_state)
                self._mps = self._quantum_state_schmidt_decomposition(quantum_state)
        else:
            self._mps = [QubitTensor(self._initial_qtensor())]
            if qubits > 1:
                for _ in range(qubits - 1):
                    self._mps.append(Normalize(self._initial_normalize()))
                    self._mps.append(QubitTensor(self._initial_qtensor()))

    def _initial_qtensor(self):
        return self._array_helper.array([1, 0], dtype=self._dtype).reshape(1, 2, 1)

    def _initial_normalize(self):
        return self._array_helper.array([1, ], dtype=self._dtype)

    def _quantum_state_schmidt_decomposition(self, quantum_state: np.ndarray) -> list:
        """ Generate the Matrix Product State from the given quantum state by the Schmidt Decomposition

        Args:
            quantum_state (np.ndarray): The given quantum state

        Returns:
            list: The MPS list
        """

        assert len(quantum_state.shape) == 1
        quantum_state = quantum_state.reshape(1, -1)
        mp_state = []
        for _ in range(self.qubits - 1):
            ldim = quantum_state.shape[0]
            quantum_state = quantum_state.reshape(2 * ldim, -1)

            U, S, VT = self._array_helper.linalg.svd(quantum_state, full_matrices=False)
            mp_state.append(QubitTensor(U.reshape(ldim, 2, -1)))
            mp_state.append(Normalize(S))
            quantum_state = VT

        mp_state.append(QubitTensor(VT.reshape(-1, 2, 1)))
        return mp_state

    def apply_single_gate(self, qubit_index: int, gate_matrix: np.ndarray):
        """ Apply single gate into MPS

        Args:
            qubit_index (int): The index of qubit
            gate_matrix (np.ndarray): The matrix of the quantum gate
        """
        assert qubit_index < self.qubits
        target_qubit = self._mps[qubit_index * 2]
        target_qubit.tensor_data = self._array_helper.tensordot(
            target_qubit.tensor_data, self._array_helper.array(gate_matrix, dtype=self._dtype), [[1], [1]]
        ).transpose([0, 2, 1])

    def apply_double_gate(self, qubit_indexes: list, gate_matrix: np.ndarray, inverse: bool = False):
        """ Apply bi-qubits quantum gate into MPS

        Args:
            qubit_indexes (list): The list of qubits' indexes
            gate_matrix (np.ndarray): The matrix of the quantum gate
            inverse (bool, optional): Need Extra inverse? only for CRx and Unitary. Defaults to False.
        """
        q0, q1 = qubit_indexes
        if abs(q0 - q1) == 1:
            # Apply consecutive bi-qubits gate
            self.apply_consecutive_double_gate(qubit_indexes, gate_matrix, inverse)
        else:
            # Apply swap gate for un-consecutive bi-qubits gate
            min_q = min(qubit_indexes)
            max_q = max(qubit_indexes)
            for i in range(min_q, max_q - 1):
                self.apply_consecutive_double_gate([i, i + 1], self.__SWAP_MATRIX)

            qubit_indexes = [max_q, max_q - 1] if q0 > q1 else [max_q - 1, max_q]
            self.apply_consecutive_double_gate(qubit_indexes, gate_matrix, inverse)
            for i in range(max_q - 1, min_q, -1):
                self.apply_consecutive_double_gate([i - 1, i], self.__SWAP_MATRIX)

    def apply_consecutive_double_gate(self, qubit_indexes: list, gate_matrix: np.ndarray, inverse: bool = False):
        """ Apply bi-qubits quantum gate with consecutive qubits' indexes into MPS.

        Args:
            qubit_indexes (list): The list of qubits' indexes
            gate_matrix (np.ndarray): The matrix of the quantum gate
            inverse (bool, optional): Need Extra inverse? only for CRx and Unitary.. Defaults to False.
        """
        if inverse:
            gate_matrix = self._linear_algorithm.MatrixPermutation(gate_matrix, np.array([1, 0]))

        if qubit_indexes[0] > qubit_indexes[1]:
            qubit_indexes.sort()
            gate_matrix = self._linear_algorithm.MatrixPermutation(gate_matrix, np.array([1, 0]))

        # Only support consecutive qubits gate
        qubit0 = self._mps[qubit_indexes[0] * 2]
        qubit1 = self._mps[qubit_indexes[1] * 2]
        norm = self._mps[qubit_indexes[1] * 2 - 1]

        # Combined two qubits together
        q0_data = qubit0.tensor_data
        q1_data = qubit1.tensor_data
        q0_ldim, q1_rdim = q0_data.shape[0], q1_data.shape[2]
        bi_qubits_comb = self._array_helper.tensordot(
            self._array_helper.tensordot(q0_data, self._array_helper.diag(norm.matrix_data), [2, 0]),
            q1_data,
            [2, 0]
        )
        bi_qubits_comb = bi_qubits_comb.reshape([q0_ldim, -1, q1_rdim])

        # Apply two qubit gate
        bi_qubits_comb = self._array_helper.tensordot(
            bi_qubits_comb,
            self._array_helper.array(gate_matrix, dtype=self._dtype),
            [[1], [1]]
        ).transpose([0, 2, 1])
        # schmidt decomposition
        bi_qubits_comb = bi_qubits_comb.reshape(q0_ldim * 2, -1)
        U, S, VT = self._array_helper.linalg.svd(bi_qubits_comb, full_matrices=False)

        # Put back to Qi, Ni, Qi+1
        qubit0.tensor_data = U.reshape(q0_ldim, 2, -1)
        qubit1.tensor_data = VT.reshape(-1, 2, q1_rdim)
        norm.matrix_data = S

    # temp
    def show(self, only_shape: bool = False):
        print(f"Qubits number: {self.qubits}.")
        idx = 0
        for site in self._mps:
            if isinstance(site, QubitTensor):
                print(f"Qubit {idx}'s tensor:")
                if not only_shape:
                    print(site.tensor_data)
                print(site.tensor_data.shape)
            else:
                print(f"Norm {idx}:")
                if not only_shape:
                    print(site.matrix_data)
                print(site.matrix_data.shape)
                idx += 1

    def to_statevector(self) -> np.ndarray:
        """ Transfer MPS into State Vector. WARNING: it will generate an vector with size 2**n, where n is the
         number of qubits.

        Returns:
            np.ndarray: The state vector from MPS
        """
        state_vector = self._mps[0].tensor_data
        for i in range(1, self.qubits):
            state_vector = self._array_helper.tensordot(
                state_vector, self._array_helper.diag(self._mps[2 * i - 1].matrix_data), [[-1], [0]]
            )
            state_vector = self._array_helper.tensordot(state_vector, self._mps[2 * i].tensor_data, [[-1], [0]])
            ldim, rdim = state_vector.shape[0], state_vector.shape[-1]
            state_vector = state_vector.reshape([ldim, -1, rdim])

        return state_vector.flatten('C')
