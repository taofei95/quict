from collections import defaultdict
import numpy as np

from QuICT.core import *
from QuICT.ops.linalg.cpu_calculator import multiply, dot, tensor, MatrixPermutation


class Optimizer:
    def __init__(self):
        self._opt_gates = []
        self._gates_by_qubit = defaultdict(list)
        self._two_qubits_opt_gates_idxes_dict = {}
        self._based_tensor_matrix = np.identity(2, dtype=np.complex128)

    def optimize(self, gates: list) -> list:
        self._cache_clear()
        special_gates = []

        # Combined the gates by qubit
        for gate in gates:
            if gate.is_special():
                special_gates.append(gate)
                continue

            if gate.is_single():
                self._gates_by_qubit[gate.targ].append(gate)
            else:
                qubit_idxes = gate.cargs + gate.targs
                gate_idx_bit = np.sum([1 << qidx for qidx in qubit_idxes])
                self.two_qubits_gates_combined(
                    gate,
                    self._gates_by_qubit[qubit_idxes[0]],
                    self._gates_by_qubit[qubit_idxes[1]],
                    gate_idx_bit,
                    reverse=qubit_idxes[0] > qubit_idxes[1]
                )

                for idx in qubit_idxes:
                    del self._gates_by_qubit[idx]

        for _, gates in self._gates_by_qubit.items():
            self.single_gates_combined(gates)

        return self._opt_gates + special_gates

    def _cache_clear(self):
        self._opt_gates = []
        self._gates_by_qubit = defaultdict(list)
        self._two_qubits_opt_gates_idxes_dict = {}

    def single_gates_combined(self, gates, matrix_only: bool = False):
        based_matrix = gates[0].compute_matrix
        is_diagonal = gates[0].is_diagonal()
        for gate in gates[1:]:
            if is_diagonal and gate.is_diagonal():
                based_matrix = multiply(gate.compute_matrix, based_matrix)
            else:
                based_matrix = dot(gate.compute_matrix, based_matrix)
                if is_diagonal:
                    is_diagonal = False

        if matrix_only:
            return based_matrix

        opt_gate = UnitaryGate()
        opt_gate.name = f"UnitaryGate_{len(self._opt_gates)}"
        opt_gate.targs = gates[0].targs
        opt_gate.targets = 1
        opt_gate.matrix = based_matrix
        opt_gate.matrix_type = "diagonal" if is_diagonal else "non-diagonal"

        self._opt_gates.append(opt_gate)

    def two_qubits_gates_combined(self, two_qubit_gate, cidx_gates, tidx_gates, gate_idx_bit, reverse: bool):
        # Combined single-qubit gates first
        if cidx_gates:
            cidx_matrix = self.single_gates_combined(cidx_gates, matrix_only=True)
        else:
            cidx_matrix = self._based_tensor_matrix

        if tidx_gates:
            tidx_matrix = self.single_gates_combined(tidx_gates, matrix_only=True)
        else:
            tidx_matrix = self._based_tensor_matrix

        combined_single_gates = tensor(cidx_matrix, tidx_matrix)
        opt_gate_matrix = dot(two_qubit_gate.compute_matrix, combined_single_gates)
        # Permutation if reverse, as it considers the sequence of qubit indexes in gate function.
        if reverse:
            opt_gate_matrix = MatrixPermutation(opt_gate_matrix, np.array([1, 0]))

        is_find = False     # find the pre-generate two qubits gate with same qubit indexes
        update_idx = []     # Update the two qubits gate dict
        for idx_bit, _ in self._two_qubits_opt_gates_idxes_dict.items():
            if idx_bit == gate_idx_bit:
                is_find = True
                break

            if idx_bit & gate_idx_bit:
                update_idx.append(idx_bit)

        for del_idx in update_idx:
            del self._two_qubits_opt_gates_idxes_dict[del_idx]

        if is_find:
            opt_gate = self._opt_gates[self._two_qubits_opt_gates_idxes_dict[gate_idx_bit]]
            opt_gate.matrix = dot(opt_gate_matrix, opt_gate.matrix)
        else:
            opt_gate = UnitaryGate()
            opt_gate.name = f"UnitaryGate_{len(self._opt_gates)}"
            opt_gate.targs = two_qubit_gate.targs
            opt_gate.cargs = two_qubit_gate.cargs
            opt_gate.targets = len(two_qubit_gate.targs)
            opt_gate.controls = len(two_qubit_gate.cargs)
            opt_gate.matrix = opt_gate_matrix

            self._two_qubits_opt_gates_idxes_dict[gate_idx_bit] = len(self._opt_gates)
            self._opt_gates.append(opt_gate)
 
    def circuit_divide(self, gates: list) -> list:
        gate_groups = defaultdict(list)
        qubit_groups = {}

        for gate in gates:
            if gate.is_single():
                continue

            gate_idx = gate.cargs + gate.targs

            gate_idx_bit = 1 << gate_idx[0] + 1 << gate_idx[1]
            idx_in_group = []

            for qubits in qubit_groups.keys():
                if qubits & gate_idx_bit:
                    idx_in_group.append(qubits)

            if len(idx_in_group) == 0:
                qubit_groups[gate_idx_bit] = 2
                gate_groups[gate_idx_bit].append(gate)
            elif len(idx_in_group) == 1:
                pre_gate_bit = idx_in_group[0]
                new_qubit_group = gate_idx_bit | pre_gate_bit
                group_qubit_num = qubit_groups[pre_gate_bit] + 1
                qubit_groups[new_qubit_group] = group_qubit_num
                gate_groups[new_qubit_group] = gate_groups[pre_gate_bit].append(gate)
                
                del gate_groups[pre_gate_bit]
                del qubit_groups[pre_gate_bit]
                
                if group_qubit_num >= 15:
                    break

            elif len(idx_in_group) == 2:
                for gate_bit in idx_in_group:
                    gate_idx_bit = gate_idx_bit | gate_bit
                
                group_qubit_num = qubit_groups[idx_in_group[0]] + qubit_groups[idx_in_group[1]]
                
                if group_qubit_num >= 15:
                    break
                else:
                    qubit_groups[gate_idx_bit] = qubit_groups[idx_in_group[0]] + qubit_groups[idx_in_group[1]]
                    gate_groups[gate_idx_bit] = gate_groups[idx_in_group[0]] + gate_groups[idx_in_group[1]] + [gate]

                for gate_bit in idx_in_group:
                    del gate_groups[gate_bit]
                    del qubit_groups[gate_bit]

        return gate_groups
