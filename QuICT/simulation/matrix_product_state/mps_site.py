import numpy as np

from.schmidt_decompostion import schmidt_decompose


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
            self._matrix_data = np.array([1,], dtype=np.complex128)


class MPSSiteStructure:
    @property
    def qubits(self) -> int:
        return self._qubits

    def __init__(
        self,
        qubits: int,
        quantum_state: np.ndarray = None,
        special_mode: str = None,
        device: str = "CPU",
        precision: str = "double"
    ):
        self._qubits = qubits
        assert device in ["CPU", "GPU"]
        self._device = device
        assert precision in ["double", "single"]
        self._precision = precision

        if quantum_state is not None:
            self._mps = self._quantum_state_schmidt_decomposition(quantum_state)

        if special_mode is None:
            self._mps = [QubitTensor()]
            if qubits > 1:
                for _ in range(qubits - 1):
                    self._mps.append(Normalize())
                    self._mps.append(QubitTensor())

    def _quantum_state_schmidt_decomposition(self, quantum_state: np.ndarray) -> list:
        # TODO: Later refactoring schmidt_decompose
        mp_state = []
        for i in range(self.qubits - 1):
            S, U, V = schmidt_decompose(quantum_state, 1)
            mp_state.append(QubitTensor(U))
            mp_state.append(Normalize(S))

            if i == self.qubits - 2:
                mp_state.append(QubitTensor(V))
            else:
                quantum_state = V.reshape(2 * V.shape[0], -1)

        return mp_state

    def apply_single_gate(self, qubit_index: int, gate_matrix: np.ndarray):
        assert qubit_index < self.qubits
        self._mps[qubit_index * 2].apply_single_gate(gate_matrix)

    def apply_double_gate(self, qubit_indexes: list, gate_matrix: np.ndarray):
        # Only support consecutive qubits gate
        qubit0 = self._mps[qubit_indexes[0] * 2]
        qubit1 = self._mps[qubit_indexes[1] * 2]
        norm = self._mps[qubit_indexes[1] * 2 - 1]

        # Combined two qubits together
        q0_data = qubit0.tensor_data
        q1_data = qubit1.tensor_data
        # if qubit_indexes[0] != 0:
        #     q0_data = np.tensordot(
        #         self._mps[qubit_indexes[0] * 2 - 1].diagonal_matrix,
        #         q0_data,
        #         [1, 0]
        #     )

        bi_qubits_comb = np.tensordot(
            np.tensordot(q0_data, norm.diagonal_matrix, [2, 0]),
            q1_data,
            [2, 0]
        )
        ldim, _, _, rdim = bi_qubits_comb.shape
        bi_qubits_comb = bi_qubits_comb.reshape([ldim, -1, rdim])
        # if qubit_indexes[1] != self.qubits - 1:
        #     bi_qubits_comb = np.tensordot(
        #         bi_qubits_comb,
        #         self._mps[qubit_indexes[1] * 2 + 1].diagonal_matrix,
        #         [2, 0]
        #     )

        # Apply two qubit gate
        bi_qubits_comb = np.tensordot(
            bi_qubits_comb,
            gate_matrix,
            [[1], [0]]
        ).transpose([0, 2, 1])

        # schmidt decomposition
        if ldim == 1 and rdim == 1:
            bi_qubits_comb = bi_qubits_comb.reshape(2, -1)
        else:
            bi_qubits_comb = bi_qubits_comb.reshape(ldim * rdim, -1)

        U, S, VT = np.linalg.svd(bi_qubits_comb, full_matrices=False)

        # Put back to Qi, Ni, Qi+1
        qubit0.tensor_data = VT.reshape(ldim, 2, -1)
        qubit1.tensor_data = U.reshape(-1, 2, rdim)
        norm.matrix_data = S

    # temp
    def show(self):
        print(f"Qubits number: {self.qubits}.")
        idx = 0
        for site in self._mps:
            if isinstance(site, QubitTensor):
                print(f"Qubit {idx}'s tensor:")
                print(site.tensor_data)
                print(site.tensor_data.shape)
            else:
                print(f"Norm {idx}:")
                print(site.matrix_data)
                idx += 1

    def to_statevector(self):
        state_vector = self._mps[0].tensor_data
        for i in range(1, self.qubits):
            state_vector = np.tensordot(state_vector, self._mps[2 * i - 1].diagonal_matrix, [[-1], [0]]) # .transpose([1, 0, 2])
            state_vector = np.tensordot(state_vector, self._mps[2 * i].tensor_data, [[-1], [0]])

        return state_vector.flatten()
