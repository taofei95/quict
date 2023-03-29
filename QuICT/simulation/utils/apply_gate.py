import numpy as np

from QuICT.core.gate import BasicGate, GateMatrixGenerator
from QuICT.core.utils import GateType, MatrixType
from QuICT.ops.utils import LinAlgLoader
from QuICT.ops.linalg.cpu_calculator import (
    matrix_dot_vector, diagonal_matrix, swap_matrix, reverse_matrix, measure_gate_apply, reset_gate_apply
)
from QuICT.tools.exception.core import GateQubitAssignedError
from QuICT.tools.exception.simulation import GateTypeNotImplementError, GateAlgorithmNotImplementError


class GateSimulator:
    def __init__(self, device, gpu_device_id: int = 0, sync: bool = True):
        self._gate_matrix_generator = GateMatrixGenerator()
        self._device = device
        self._gpu_device_id = gpu_device_id
        self._sync = sync
        if self._device == "GPU":
            import cupy as cp

            self._array_helper = cp
            self._algorithm = LinAlgLoader(device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=False)

    def _get_gate_matrix(self, gate: BasicGate):
        if self._device == "CPU":
            return self._gate_matrix_generator.get_matrix(gate)
        else:
            return self._gate_matrix_generator.get_matrix(gate, special_array_generator=self._array_helper)

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
            return self._apply_gate_cpu(gate, gate_cargs, gate_targs, state_vector, qubits)
        else:
            return self._apply_gate_gpu(gate, gate_cargs, gate_targs, state_vector, qubits)

    def _apply_gate_cpu(
        self,
        gate: BasicGate,
        cargs: list,
        targs: list,
        state_vector,
        qubits
    ):
        matrix_type = gate.matrix_type
        args_num = gate.controls + gate.targets
        control_idx = np.array(cargs, dtype=np.int64)
        target_idx = np.array(targs, dtype=np.int64)
        default_params = (
            state_vector, qubits, gate.matrix, args_num, control_idx, target_idx
        )

        if matrix_type in [MatrixType.diag_diag, MatrixType.diagonal, MatrixType.control]:
            diagonal_matrix(
                *default_params,
                is_control=True if matrix_type == MatrixType.control else False
            )
        elif matrix_type == MatrixType.swap:
            swap_matrix(*default_params)
        elif matrix_type == MatrixType.reverse:
            reverse_matrix(*default_params)
        else:
            matrix_dot_vector(
                state_vector,
                qubits,
                gate.matrix,
                gate.controls + gate.targets,
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
        matrix = self._get_gate_matrix(gate)

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
            self.apply_control_matrix(matrix[-1, -1].get(), args_num, cargs, targs, state_vector, qubits)
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

    def apply_measure_gate(self, index: int, state_vector: np.ndarray, qubits: int) -> int:
        if self._device == "CPU":
            result = measure_gate_apply(index, state_vector)
        else:
            prob = self._algorithm.measured_prob_calculate(
                index, state_vector, qubits, sync=self._sync
            )
            result = int(self._algorithm.apply_measuregate(
                index, state_vector, qubits, prob, self._sync
            ))

        return result

    def apply_reset_gate(self, index: int, state_vector: np.ndarray, qubits: int):
        if self._device == "CPU":
            reset_gate_apply(index, state_vector)
        else:
            prob = self._algorithm.measured_prob_calculate(
                index, state_vector, qubits, sync=self._sync
            )
            self._algorithm.apply_resetgate(
                index, state_vector, qubits, prob, self._sync
            )

    def get_measured_prob(self, index: int, state_vector: np.ndarray, qubits: int):
        if self._device == "CPU":
            return get_measured_probability(index, state_vector)
        else:
            return self._algorithm.measured_prob_calculate(
                index, state_vector, qubits, sync=self._sync
            )
