from typing import *
import numpy as np

from .gate_type import MatrixType
import QuICT.ops.linalg.cpu_calculator as CPUCalculator


class CircuitMatrix:
    def __init__(self, device: str = "CPU"):
        self._device = device
        if device == "CPU":
            self._computer = CPUCalculator
            self._array_helper = np
        else:
            import cupy as cp
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator

            self._computer = GPUCalculator
            self._array_helper = cp

    def get_unitary_matrix(self, gates: list, qubits_num: int) -> np.ndarray:
        matrix_groups = []          # List[List[(gate.matrix, gate.args)]]
        inside_qubits = {}          # Dict[qubit_idx: related index in matrix_groups]
        depth_qubits = [0] * qubits_num
        for gate in gates:
            if gate.controls + gate.targets >= 3:
                raise Exception("only support 2-qubit gates and 1-qubit gates.")

            if gate.matrix_type == MatrixType.special:
                continue

            args = gate.cargs + gate.targs
            matrix = gate.matrix if self._device == "CPU" else self._array_helper.array(gate.matrix)
            if len(args) == 1:      # Deal with single-qubit gate
                if args[0] in inside_qubits.keys():
                    related_matrix_group_idx = inside_qubits[args[0]]
                    matrix_groups[related_matrix_group_idx].append((matrix, args))
                else:
                    matrix_groups.append([(matrix, args)])
                    inside_qubits[args[0]] = len(matrix_groups) - 1
            else:       # Deal with double-qubits gate
                if args[0] > args[1]:
                    args.sort()
                    matrix = self._computer.MatrixPermutation(matrix, self._array_helper.array([1, 0]))

                intersect_qubits = list(set(args) & set(inside_qubits.keys()))
                related_matrix_group_idx = len(matrix_groups)
                if len(intersect_qubits) == 0:
                    matrix_groups.append([(matrix, args)])
                elif len(intersect_qubits) == 1:
                    if depth_qubits[args[0]] == depth_qubits[args[1]]:
                        related_matrix_group_idx = inside_qubits[intersect_qubits[0]]
                        matrix_groups[related_matrix_group_idx].append((matrix, args))
                    else:
                        matrix_groups.append([(matrix, args)])
                        related_idx = inside_qubits[intersect_qubits[0]]
                        updated_depth = max(depth_qubits[args[0]], depth_qubits[args[1]])
                        covered_qubits = [qid for qid, rid in inside_qubits.items() if rid == related_idx]
                        for c_q in covered_qubits:
                            del inside_qubits[c_q]
                            depth_qubits[c_q] = updated_depth
                else:
                    related_idx0 = inside_qubits[args[0]]
                    related_idx1 = inside_qubits[args[1]]
                    if related_idx0 == related_idx1:
                        matrix_groups[related_idx0].append((matrix, args))
                        continue
                    else:
                        matrix_groups.append([(matrix, args)])
                        covered_qubits = [qid for qid, rid in inside_qubits.items() if rid == related_idx0 or rid == related_idx1]
                        for c_q in covered_qubits:
                            del inside_qubits[c_q]
                            depth_qubits[c_q] += 1

                for arg in args:
                    inside_qubits[arg] = related_matrix_group_idx

        combined_matries = []       # List[gate]
        for glist in matrix_groups:
            combined_matries.append(self._combined_gates(glist))

        return self._combined_gates(combined_matries)

    def merge_gates(self, u1, u1_args, u2, u2_args):
        u1_args_set, u2_args_set = set(u1_args), set(u2_args)
        insection_args = u1_args_set & u2_args_set
        if len(insection_args) == 0:
            return self._computer.tensor(u1, u2), u1_args + u2_args

        if u1_args_set == u2_args_set:
            args_idx = [u1_args.index(u2_arg) for u2_arg in u2_args]
            u2 = self._computer.MatrixPermutation(
                u2,
                self._array_helper.array(args_idx)
            )

        union_args = u1_args + [i for i in u2_args if i not in u1_args] if len(u1_args) >= len(u2_args) else \
            u2_args + [i for i in u1_args if i not in u2_args]
        if len(union_args) != len(u1_args_set):
            u1 = self._tensor_unitary(u1, u1_args, union_args)

        if len(union_args) != len(u2_args_set):
            u2 = self._tensor_unitary(u2, u2_args, union_args)

        return self._computer.dot(u2, u1), union_args

    def _tensor_unitary(self, unitary, unitary_args, extend_args):
        uargs_idx = [extend_args.index(uarg) for uarg in unitary_args]
        mono_diff = np.diff(uargs_idx)
        if np.allclose(abs(mono_diff), 1):
            if np.allclose(mono_diff, -1):
                unitary = self._computer.MatrixPermutation(
                    unitary,
                    self._array_helper.arange(len(unitary_args) - 1, -1, -1)
                )

            return self._computer.MatrixTensorI(
                unitary,
                1 << (min(uargs_idx) - 0),
                1 << (len(extend_args) - 1 - max(uargs_idx))
            )

        tensor_matrix = self._computer.MatrixTensorI(
            unitary,
            1,
            1 << (len(extend_args) - len(unitary_args))
        )
        tmatrix_args = unitary_args + [earg for earg in extend_args if earg not in unitary_args]
        permutation_index = [extend_args.index(tm_arg) for tm_arg in tmatrix_args]

        return self._computer.MatrixPermutation(tensor_matrix, self._array_helper.array(permutation_index))

    def _combined_gates_by_order(self, gates):
        based_matrix, based_matrix_args = gates[0]
        for i in range(1, len(gates)):
            current_matrix, current_matrix_args = gates[i]
            based_matrix, based_matrix_args = self.merge_gates(
                based_matrix, based_matrix_args,
                current_matrix, current_matrix_args
            )

        return based_matrix, based_matrix_args

    def _combined_gates(self, gates):
        args_num = [len(args) for _, args in gates]
        while len(args_num) > 1:
            left, right = self._find_mini_args_num(args_num)
            lg, rg = gates[left], gates[right]
            combined_matrix, combined_matrix_args = self.merge_gates(
                lg[0], lg[1],
                rg[0], rg[1]
            )
            gates[left] = (combined_matrix, combined_matrix_args)
            del gates[right]

            args_num[left] += args_num[right]
            del args_num[right]

        return gates[0]

    def _find_mini_args_num(self, nums: list):
        mini_num = sum(nums)*2
        left, right = -1, -1
        for i in range(1, len(nums), 1):
            if nums[i] + nums[i - 1] < mini_num:
                left = i - 1
                right = i
                mini_num = nums[i] + nums[i - 1]

        return left, right
