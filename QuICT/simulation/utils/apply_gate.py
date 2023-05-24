import numpy as np

from QuICT.core.gate import BasicGate, GateMatrixGenerator
from QuICT.core.utils import GateType, MatrixType, matrix_product_to_circuit
from QuICT.ops.utils import LinAlgLoader
from QuICT.tools.exception.core import GateQubitAssignedError, ValueError
from QuICT.tools.exception.simulation import GateTypeNotImplementError, GateAlgorithmNotImplementError


class GateSimulator:
    def __init__(self, device, precision: str = "double", gpu_device_id: int = 0, sync: bool = True):
        if device not in ["CPU", "GPU"]:
            raise ValueError("Simulation.device", "[CPU, GPU]", device)

        if precision not in ["single", "double"]:
            raise ValueError("Simulation.precision", "[single, double]", precision)

        self._device = device
        self._precision = precision
        self._dtype = np.complex128 if precision == "double" else np.complex64
        self._gpu_device_id = gpu_device_id
        self._sync = sync

        # Gate's Matrix Store
        self._gate_matrix_generator = GateMatrixGenerator()
        self._gate_matrix_info: dict = {}   # Record gate's matrix info (location, length)
        self._gates_matrix = None

        # cupy/numpy and Gate Kernel functions
        self._algorithm = LinAlgLoader(device=device, enable_gate_kernel=True)
        if self._device == "GPU":
            import cupy as cp

            self._array_helper = cp
            cp.cuda.runtime.setDevice(self._gpu_device_id)
        else:
            self._array_helper = np

    ####################################################################
    ############          State Vector Generator            ############
    ####################################################################
    def get_empty_state_vector(self, qubits: int):
        return self._array_helper.zeros(1 << qubits, dtype=self._dtype)

    def get_allzero_state_vector(self, qubits: int):
        state_vector = self.get_empty_state_vector(qubits)
        state_vector[0] = self._dtype(1)

        return state_vector

    def normalized_state_vector(self, state_vector, qubits: int):
        assert 1 << qubits == state_vector.size, "The state vector should has the same qubits with the circuit."
        if not type(state_vector) is self._array_helper.ndarray:
            state_vector = self._array_helper.array(state_vector, dtype=self._dtype)

        if state_vector.dtype != self._dtype:
            state_vector = state_vector.astype(self._dtype)

        return state_vector

    ####################################################################
    ############          State Vector Generator            ############
    ####################################################################
    def get_allzero_density_matrix(self, qubits: int):
        density_matrix = self._array_helper.zeros((1 << qubits, 1 << qubits), dtype=self._dtype)
        density_matrix[0, 0] = self._dtype(1)

        return density_matrix

    def get_empty_density_matrix(self, qubits: int):
        return self._array_helper.zeros((1 << qubits, 1 << qubits), dtype=self._dtype)

    def validate_density_matrix(self, matrix) -> bool:
        """ Density Matrix Validation. """
        if not isinstance(matrix, np.ndarray):
            matrix = matrix.get()

        if not np.allclose(matrix.T.conjugate(), matrix):
            raise ValueError("The conjugate transpose of density matrix do not equal to itself.")

        eigenvalues = np.linalg.eig(matrix)[0]
        for ev in eigenvalues:
            if ev < 0 and not np.isclose(ev, 0, rtol=1e-4):
                raise ValueError("The eigenvalues of density matrix should be non-negative")

        if not np.isclose(matrix.trace(), 1, rtol=1e-4):
            raise ValueError("The sum of trace of density matrix should be 1.")

        return True

    ####################################################################
    ############           Gate Matrix Generator            ############
    ####################################################################
    def normalized_matrix(self, unitary_matrix, qubits: int):
        row, col = unitary_matrix.shape
        assert 1 << qubits == row and row == col, "The unitary matrix should be square."
        if not type(unitary_matrix) is self._array_helper.ndarray:
            unitary_matrix = self._array_helper.array(unitary_matrix, dtype=self._dtype)

        if unitary_matrix.dtype != self._dtype:
            unitary_matrix = unitary_matrix.astype(self._dtype)

        return unitary_matrix

    def is_identity(self, unitary_matrix):
        row = unitary_matrix.shape[0]
        identity_matrix = self._array_helper.identity(row, dtype=self._dtype)
        return self._array_helper.allclose(unitary_matrix, identity_matrix)

    def dot(self, unitary_matrix, state_vector):
        if self.is_identity(unitary_matrix):
            return state_vector
        else:
            return self._algorithm.dot(unitary_matrix, state_vector)

    def gate_matrix_combined(self, circuit):
        matrix_list_for_gpu_only = []
        total_matrix_size = 0 if self._gates_matrix is None else self._gates_matrix.size
        for gate in circuit.flatten_gates():
            if (
                not isinstance(gate, BasicGate)
                or gate.type in [GateType.measure, GateType.reset, GateType.barrier, GateType.unitary]
            ):
                continue

            gate_name = self._generate_gate_name_for_matrix_stored(gate.type, gate.pargs)
            if gate_name not in self._gate_matrix_info.keys():
                matrix = self._gate_matrix_generator.get_matrix(gate, self._precision)
                if self._device == "CPU":
                    self._gate_matrix_info[gate_name] = matrix
                else:
                    self._gate_matrix_info[gate_name] = (total_matrix_size, matrix.size)
                    total_matrix_size += matrix.size
                    matrix_list_for_gpu_only.append(matrix)

        if len(matrix_list_for_gpu_only) > 0:
            self._concentrate_gate_matrixs(matrix_list_for_gpu_only, total_matrix_size)

    def _generate_gate_name_for_matrix_stored(self, gate_type, gate_pargs: list = None):
        gate_name = str(gate_type)

        for parg in gate_pargs:
            gate_name += f"_{parg}"

        return gate_name

    def _concentrate_gate_matrixs(self, matrix_list: list, matrix_size: int):
        gates_matrix = np.empty(matrix_size, dtype=self._dtype)
        start = 0
        for matrix in matrix_list:
            gates_matrix[start:start + matrix.size] = matrix.ravel()[:]
            start += matrix.size

        gates_matrix = self._array_helper.array(gates_matrix)
        if self._gates_matrix is None:
            self._gates_matrix = gates_matrix
        else:
            self._gates_matrix = self._array_helper.concatenate((self._gates_matrix, gates_matrix))

    def _get_gate_matrix(self, gate: BasicGate):
        gate_name = self._generate_gate_name_for_matrix_stored(gate.type, gate.pargs)

        if self._device == "CPU":
            return self._gate_matrix_info[gate_name]
        else:
            start, interval = self._gate_matrix_info[gate_name]
            return self._gates_matrix[start:start + interval]

    ####################################################################
    ############           Gate Kernel Functions            ############
    ####################################################################
    def apply_gate(
        self,
        gate: BasicGate,
        assigned_qubits: list,
        state_vector: np.ndarray,
        qubits: int
    ):
        gate_type = gate.type
        if (
            gate_type in [GateType.id, GateType.barrier] or
            gate.is_identity()
        ):
            return

        gate_cargs = [qubits - 1 - assigned_qubits[i] for i in range(gate.controls)]
        gate_targs = [qubits - 1 - assigned_qubits[i] for i in range(gate.controls, len(assigned_qubits))]
        if self._device == "CPU":
            return self._apply_gate_cpu(gate, gate_cargs, gate_targs, state_vector)
        else:
            return self._apply_gate_gpu(gate, gate_cargs, gate_targs, state_vector, qubits)

    def _apply_gate_cpu(
        self,
        gate: BasicGate,
        cargs: list,
        targs: list,
        state_vector
    ):
        matrix_type = gate.matrix_type
        matrix = self._get_gate_matrix(gate) if gate.type != GateType.unitary else gate.matrix
        control_idx = np.array(cargs, dtype=np.int64)
        target_idx = np.array(targs, dtype=np.int64)

        if matrix_type in [MatrixType.diag_diag, MatrixType.diagonal, MatrixType.control]:
            self._algorithm.diagonal_matrix(
                state_vector,
                matrix,
                control_idx,
                target_idx,
                is_control=True if matrix_type == MatrixType.control else False
            )
        elif matrix_type == MatrixType.swap:
            self._algorithm.swap_matrix(state_vector, matrix, control_idx, target_idx)
        elif matrix_type == MatrixType.reverse:
            self._algorithm.reverse_matrix(state_vector, matrix, control_idx, target_idx)
        else:
            self._algorithm.matrix_dot_vector(
                state_vector,
                matrix,
                np.append(target_idx, control_idx)
            )

    def _apply_gate_gpu(
        self,
        gate: BasicGate,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        gate_type, matrix_type = gate.type, gate.matrix_type
        args_num = gate.controls + gate.targets
        matrix = self._get_gate_matrix(gate) if gate.type != GateType.unitary else self._array_helper.array(gate.matrix)

        # Deal with quantum gate with more than 3 qubits.
        if gate_type == GateType.unitary and args_num >= 3:
            state_vector = self._algorithm.matrix_dot_vector(
                state_vector,
                qubits,
                matrix,
                cargs + targs,
                self._sync
            )

            return

        # [H, Hy, SX, SY, SW, U2, U3, Rx, Ry] 2-bits [CH, ] 2-bits[Rzx, targets, unitary]
        if matrix_type == MatrixType.normal:
            self.apply_normal_matrix(matrix, args_num, cargs, targs, state_vector, qubits)
        # [Rz, Phase], 2-bits [CRz, Rzz], 3-bits [CCRz]
        elif matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]:
            self.apply_diagonal_matrix(matrix, args_num, cargs, targs, state_vector, qubits)
        # [X] 2-bits [swap, iswap, iswapdg, sqiswap] 3-bits [CSWAP]
        elif matrix_type == MatrixType.swap:
            self.apply_swap_matrix(matrix, args_num, cargs, targs, state_vector, qubits)
        # [Y] 2-bits [CX, CY] 3-bits: [CCX]
        elif matrix_type == MatrixType.reverse:
            self.apply_reverse_matrix(matrix, args_num, cargs, targs, state_vector, qubits)
        # [S, sdg, Z, U1, T, tdg] # 2-bits [CZ, CU1]
        elif matrix_type == MatrixType.control:
            self.apply_control_matrix(matrix[-1].get(), args_num, cargs, targs, state_vector, qubits)
        # [FSim]
        elif matrix_type == MatrixType.ctrl_normal:
            self._algorithm.ctrl_normal_targs(
                targs, matrix, state_vector, qubits, self._sync
            )
        # [Rxx, Ryy]
        elif matrix_type == MatrixType.normal_normal:
            self._algorithm.normal_normal_targs(
                targs, matrix, state_vector, qubits, self._sync
            )
        # [Rzx]
        elif matrix_type == MatrixType.diag_normal:
            self._algorithm.diagonal_normal_targs(
                targs, matrix, state_vector, qubits, self._sync
            )
        # [Perm] TODO: replace by build_gate
        elif gate_type == GateType.perm:
            args = cargs + targs
            if len(args) == qubits:
                mapping = np.array(gate.pargs, dtype=np.int32)
            else:
                mapping = np.arange(qubits, dtype=np.int32)
                for idx, parg in enumerate(gate.pargs):
                    mapping[args[idx]] = args[parg]

            self._algorithm.VectorPermutation(
                state_vector,
                mapping,
                changeInput=True,
                gpu_out=False,
                sync=self._sync
            )
        # unsupported quantum gates
        else:
            raise GateTypeNotImplementError(f"Unsupported Gate Type and Matrix Type: {gate_type} {matrix_type}.")

    def apply_normal_matrix(
        self,
        matrix,
        args_num: int,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        # Deal with 1-qubit normal gate e.g. H
        if args_num == 1:
            self._algorithm.normal_targ(
                targs[0], matrix, state_vector, qubits, self._sync
            )
        elif args_num == 2:     # Deal with 2-qubits control normal gate e.g. CH
            if len(cargs) == 1:
                self._algorithm.normal_ctargs(
                    cargs[0], targs[0], matrix, state_vector, qubits, self._sync
                )
            elif len(targs) == 2:     # Deal with 2-qubits unitary gate
                self._algorithm.normal_targs(
                    targs, matrix, state_vector, qubits, self._sync
                )
            else:
                raise GateQubitAssignedError("Quantum gate cannot only have control qubits.")

    def apply_diagonal_matrix(
        self,
        matrix,
        args_num: int,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        # Deal with 1-qubit diagonal gate, e.g. Rz
        if args_num == 1:
            self._algorithm.diagonal_targ(
                targs[0], matrix, state_vector, qubits, self._sync
            )
        elif args_num == 2:     # Deal with 2-qubit diagonal gate, e.g. CRz
            if len(cargs) == 1:
                self._algorithm.diagonal_ctargs(
                    cargs[0], targs[0], matrix, state_vector, qubits, self._sync
                )
            elif len(targs) == 2:
                self._algorithm.diagonal_targs(
                    targs, matrix, state_vector, qubits, self._sync
                )
            else:
                raise GateQubitAssignedError("Quantum gate cannot only have control qubits.")
        else:   # [CCRz]
            self._algorithm.diagonal_more(
                cargs, targs[0], matrix, state_vector, qubits, self._sync
            )

    def apply_swap_matrix(
        self,
        matrix,
        args_num: int,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        if args_num == 1:       # Deal with X Gate
            self._algorithm.swap_targ(
                targs[0], state_vector, qubits, self._sync
            )
        elif args_num == 2:     # Deal with Swap Gate
            self._algorithm.swap_targs(
                targs, matrix, state_vector, qubits, self._sync
            )
        else:   # CSwap
            self._algorithm.swap_tmore(
                targs, cargs[0], state_vector, qubits, self._sync
            )

    def apply_reverse_matrix(
        self,
        matrix,
        args_num: int,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        if args_num == 1:   # Deal with 1-qubit reverse gate, e.g. Y
            self._algorithm.reverse_targ(
                targs[0], matrix, state_vector, qubits, self._sync
            )
        elif args_num == 2:   # only consider 1 control qubit + 1 target qubit
            self._algorithm.reverse_ctargs(
                cargs[0], targs[0], matrix, state_vector, qubits, self._sync
            )
        else:   # CCX
            self._algorithm.reverse_more(
                cargs, targs[0], state_vector, qubits, self._sync
            )

    def apply_control_matrix(
        self,
        value,
        args_num: int,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        if args_num == 1:       # Deal with 1-qubit control gate, e.g. S
            self._algorithm.control_targ(
                targs[0], value, state_vector, qubits, self._sync
            )
        elif args_num == 2:     # Deal with 2-qubit control gate, e.g. CZ
            self._algorithm.control_ctargs(
                cargs[0], targs[0], value, state_vector, qubits, self._sync
            )
        else:
            raise GateAlgorithmNotImplementError(f"Unsupportted 3-qubits+ control unitary gate.")

    ####################################################################
    ############           Measure/Reset Function           ############
    ####################################################################
    def apply_measure_gate(self, index: int, state_vector: np.ndarray, qubits: int) -> int:
        if self._device == "CPU":
            result = int(self._algorithm.measure_gate_apply(index, state_vector))
        else:
            prob = self._algorithm.measured_prob_calculate(
                index, state_vector, qubits, sync=self._sync
            ).get()
            result = int(self._algorithm.apply_measuregate(
                index, state_vector, qubits, prob, self._sync
            ))

        return result

    def apply_reset_gate(self, index: int, state_vector: np.ndarray, qubits: int):
        if self._device == "CPU":
            self._algorithm.reset_gate_apply(index, state_vector)
        else:
            prob = self._algorithm.measured_prob_calculate(
                index, state_vector, qubits, sync=self._sync
            )
            self._algorithm.apply_resetgate(
                index, state_vector, qubits, prob, self._sync
            )

    def get_measured_prob(self, index: int, state_vector: np.ndarray, qubits: int):
        if self._device == "CPU":
            return self._algorithm.get_measured_probability(index, state_vector)
        else:
            return self._algorithm.measured_prob_calculate(
                index, state_vector, qubits, sync=self._sync
            )

    def apply_measure_gate_for_dm(self, index: int, density_matrix: np.ndarray, qubits: int):
        P0 = self._array_helper.array([[1, 0], [0, 0]], dtype=self._dtype)
        mea_0 = matrix_product_to_circuit(P0, index, qubits, self._device)
        prob_0 = self._array_helper.matmul(mea_0, density_matrix).trace()

        _1 = np.random.random() > prob_0
        if not _1:
            U = self._array_helper.matmul(
                mea_0,
                self._array_helper.eye(1 << qubits, dtype=self._dtype) / self._array_helper.sqrt(prob_0)
            )
            density_matrix = self._algorithm.dot(self._algorithm.dot(U, density_matrix), U.conj().T)
        else:
            P1 = self._array_helper.array([[0, 0], [0, 1]], dtype=self._dtype)
            mea_1 = matrix_product_to_circuit(P1, index, qubits, self._device)

            U = self._array_helper.matmul(
                mea_1,
                self._array_helper.eye(1 << qubits, dtype=self._dtype) / self._array_helper.sqrt(1 - prob_0)
            )
            density_matrix = self._algorithm.dot(self._algorithm.dot(U, density_matrix), U.conj().T)

        return _1, density_matrix
