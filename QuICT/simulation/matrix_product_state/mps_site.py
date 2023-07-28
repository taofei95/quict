import numpy as np
from typing import Union

from QuICT.ops.linalg.cpu_calculator import MatrixPermutation


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

    def __init__(self, state: np.ndarray = None):
        if state is not None:
            self._tensor_data = state
        else:
            self._tensor_data = self._origin_state()

    def _origin_state(self):
        return np.array([1, 0], dtype=np.complex128).reshape(1, 2, 1)

    def apply_single_gate(self, gate_matrix: np.ndarray):
        self._tensor_data = np.tensordot(self._tensor_data, gate_matrix, [[1], [1]]).transpose([0, 2, 1])


class Normalize:
    @property
    def matrix_data(self) -> np.ndarray:
        return self._matrix_data

    @matrix_data.setter
    def matrix_data(self, data):
        self._matrix_data = data

    @property
    def diagonal_matrix(self) -> np.ndarray:
        return np.diag(self._matrix_data)

    def __init__(self, norm_coeff: np.ndarray = None):
        if norm_coeff is not None:
            self._matrix_data = norm_coeff
        else:
            self._matrix_data = np.array([1, ], dtype=np.complex128)


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
        assert device in ["CPU", "GPU"]
        self._device = device
        assert precision in ["double", "single"]
        self._precision = precision

    def initial_mps(self, qubits: int, quantum_state: Union[str, np.ndarray] = None):
        self._qubits = qubits
        if quantum_state is not None:
            if not isinstance(quantum_state, str):
                self._mps = self._quantum_state_schmidt_decomposition(quantum_state)
        else:
            self._mps = [QubitTensor()]
            if qubits > 1:
                for _ in range(qubits - 1):
                    self._mps.append(Normalize())
                    self._mps.append(QubitTensor())

    def _quantum_state_schmidt_decomposition(self, quantum_state: np.ndarray) -> list:
        assert len(quantum_state.shape) == 1
        quantum_state = quantum_state.reshape(1, -1)
        mp_state = []
        for _ in range(self.qubits - 1):
            ldim = quantum_state.shape[0]
            quantum_state = quantum_state.reshape(2 * ldim, -1)

            U, S, VT = np.linalg.svd(quantum_state, full_matrices=False)
            mp_state.append(QubitTensor(U.reshape(ldim, 2, -1)))
            mp_state.append(Normalize(S))
            quantum_state = VT

        mp_state.append(QubitTensor(VT.reshape(-1, 2, 1)))
        return mp_state

    def apply_single_gate(self, qubit_index: int, gate_matrix: np.ndarray):
        assert qubit_index < self.qubits
        self._mps[qubit_index * 2].apply_single_gate(gate_matrix)

    def apply_double_gate(self, qubit_indexes: list, gate_matrix: np.ndarray, inverse: bool = False):
        q0, q1 = qubit_indexes
        if abs(q0 - q1) == 1:
            self.apply_consecutive_double_gate(qubit_indexes, gate_matrix, inverse)
        else:
            min_q = min(qubit_indexes)
            max_q = max(qubit_indexes)
            for i in range(min_q, max_q - 1):
                self.apply_consecutive_double_gate([i, i + 1], self.__SWAP_MATRIX)

            qubit_indexes = [max_q, max_q - 1] if q0 > q1 else [max_q - 1, max_q]
            self.apply_consecutive_double_gate(qubit_indexes, gate_matrix, inverse)
            for i in range(max_q - 1, min_q, -1):
                self.apply_consecutive_double_gate([i - 1, i], self.__SWAP_MATRIX)

    def apply_consecutive_double_gate(self, qubit_indexes: list, gate_matrix: np.ndarray, inverse: bool = False):
        if inverse:
            gate_matrix = MatrixPermutation(gate_matrix, np.array([1, 0]))

        if qubit_indexes[0] > qubit_indexes[1]:
            qubit_indexes.sort()
            gate_matrix = MatrixPermutation(gate_matrix, np.array([1, 0]))

        # Only support consecutive qubits gate
        qubit0 = self._mps[qubit_indexes[0] * 2]
        qubit1 = self._mps[qubit_indexes[1] * 2]
        norm = self._mps[qubit_indexes[1] * 2 - 1]

        # Combined two qubits together
        q0_data = qubit0.tensor_data
        q1_data = qubit1.tensor_data
        q0_ldim, q1_rdim = q0_data.shape[0], q1_data.shape[2]
        bi_qubits_comb = np.tensordot(
            np.tensordot(q0_data, norm.diagonal_matrix, [2, 0]),
            q1_data,
            [2, 0]
        )
        bi_qubits_comb = bi_qubits_comb.reshape([q0_ldim, -1, q1_rdim])

        # Apply two qubit gate
        bi_qubits_comb = np.tensordot(
            bi_qubits_comb,
            gate_matrix,
            [[1], [1]]
        ).transpose([0, 2, 1])
        # schmidt decomposition
        bi_qubits_comb = bi_qubits_comb.reshape(q0_ldim * 2, -1)
        U, S, VT = np.linalg.svd(bi_qubits_comb, full_matrices=False)

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
                print(site.diagonal_matrix.shape)
                idx += 1

    def to_statevector(self):
        state_vector = self._mps[0].tensor_data
        for i in range(1, self.qubits):
            state_vector = np.tensordot(state_vector, self._mps[2 * i - 1].diagonal_matrix, [[-1], [0]])
            state_vector = np.tensordot(state_vector, self._mps[2 * i].tensor_data, [[-1], [0]])
            ldim, rdim = state_vector.shape[0], state_vector.shape[-1]
            state_vector = state_vector.reshape([ldim, -1, rdim])

        return state_vector.flatten('C')
