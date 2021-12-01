from collections import defaultdict
import numpy as np

from QuICT.core import *
from QuICT.ops.linalg.cpu_calculator import multiply, dot, tensor, MatrixPermutation


class Optimizer:
    """ The quantum circuit optimizer, used to optimize the quantum circuit
    before simulation. Currently, it only working with the 1-, 2-qubits quantum
    gates.

    The rules:
        1. For the consecutive 1-qubit gates, it will be combined into a new 1-qubit Unitary
        gate.
        2. Merge the 1-qubit gates into the connected 2-qubits gate, if they have same qubit.
        3. If any 2-qubits gate has same target and control qubit indexes, and no other gates
        between them, merge its into a new 2-qubits Unitary gate.
    """
    def __init__(self):
        self._opt_gates = []    # The gates list after optimizer
        self._gates_by_qubit = defaultdict(list)    # Temporarily store the 1-qubit gates by qubits
        self._two_qubits_opt_gates_idxes_dict = {}  # Record the qubit indexes of the 2-qubits gates
        self._based_tensor_matrix = np.identity(2, dtype=np.complex128)

    def optimize(self, gates: list) -> list:
        """ Depending on the optimized rules, generate the optimized gates by the given quantum gates.

        Args:
            gates (list[Gate]): The list of gates before optimization

        Returns:
            list: The list of the optimized gates
        """
        self._cache_clear()

        for gate in gates:
            if gate.is_special():   # Do not optimized any special quantum gates
                self._opt_gates.append(gate)
                continue

            # The qubit indexes of the gate
            qubit_idxes = gate.cargs + gate.targs

            if len(qubit_idxes) == 1:       # 1-qubit gate
                self._gates_by_qubit[gate.targ].append(gate)
            elif len(qubit_idxes) == 2:     # 2-qubits gate
                gate_idx_bit = np.sum([1 << qidx for qidx in qubit_idxes])
                self.two_qubits_gates_combined(
                    gate,
                    self._gates_by_qubit[qubit_idxes[0]],
                    self._gates_by_qubit[qubit_idxes[1]],
                    gate_idx_bit,
                    qubit_idxes[0] > qubit_idxes[1]
                )

                for idx in qubit_idxes:
                    del self._gates_by_qubit[idx]
            else:   # TODO: not support the gates with more than 2 qubits
                self._opt_gates.append(gate)
                continue

        # Merge the rest 1-qubit gates into a Unitary gate.
        for _, gates in self._gates_by_qubit.items():
            self.single_gates_combined(gates)

        return self._opt_gates

    def _cache_clear(self):
        """ Initial the Optimizer """
        self._opt_gates = []
        self._gates_by_qubit = defaultdict(list)
        self._two_qubits_opt_gates_idxes_dict = {}

    def single_gates_combined(self, gates, matrix_only: bool = False):
        """ The optimized algorithm for 1-qubit gates with same qubit.

        Args:
            gates ([Gate]): The 1-qubit gates in the same qubit.
            matrix_only (bool, optional): return the Unitary gate or matrix. Defaults to False.

        Returns:
            (UnitaryGate, np.array): the new Unitary gate or the gate matrix
        """
        if len(gates) == 1 and not matrix_only:
            self._opt_gates.append(gates[0])
            return

        based_matrix = gates[0].compute_matrix
        is_diagonal = gates[0].is_diagonal()

        for gate in gates[1:]:
            if is_diagonal and gate.is_diagonal():  # Using multiply for diagonal gates
                based_matrix = multiply(gate.compute_matrix, based_matrix)
            else:   # Using dot for non-diagonal gates
                based_matrix = dot(gate.compute_matrix, based_matrix)
                if is_diagonal:
                    is_diagonal = False

        if matrix_only:
            return based_matrix, is_diagonal

        # Add the optimized quantum gate
        opt_gate = UnitaryGate()
        opt_gate.name = f"UnitaryGate_{len(self._opt_gates)}"
        opt_gate.targs = gates[0].targs
        opt_gate.targets = 1
        opt_gate.matrix = based_matrix
        opt_gate.diagonal = is_diagonal

        self._opt_gates.append(opt_gate)

    def two_qubits_gates_combined(self, two_qubit_gate, cidx_gates: list, tidx_gates: list, gate_idx_bit: int, reverse):
        """ The optimized algorithm for 2-qubits gates and related 1-qubit gates.

        Args:
            two_qubit_gate (Gate): The 2-qubits quantum gate
            cidx_gates (gates, None): The 1-qubit quantum gates in the control index
            tidx_gates (gates, None): The 1-qubit quantum gates in the target index
            gate_idx_bit (int): The bit-represent for the indexes of the 2-qubits quantum gate
        """
        if not cidx_gates and not tidx_gates:
            self._opt_gates.append(two_qubit_gate)
            return

        # Combined single-qubit gates first
        cm_diag, tm_diag = True, True       # diagonal matrix signal
        if cidx_gates:
            cidx_matrix, cm_diag = self.single_gates_combined(cidx_gates, matrix_only=True)
        else:
            cidx_matrix = self._based_tensor_matrix

        if tidx_gates:
            tidx_matrix, tm_diag = self.single_gates_combined(tidx_gates, matrix_only=True)
        else:
            tidx_matrix = self._based_tensor_matrix

        # Combined the gate matrix of the control and target indexes into the 2-qubits gate matrix
        combined_single_gates = tensor(cidx_matrix, tidx_matrix)

        # Combined the gate matrix of the 2-qubits gate and the merged one
        is_diagonal = (cm_diag and tm_diag and two_qubit_gate.is_diagonal())
        if is_diagonal:
            opt_gate_matrix = multiply(two_qubit_gate.compute_matrix, combined_single_gates)
        else:
            opt_gate_matrix = dot(two_qubit_gate.compute_matrix, combined_single_gates)

        if reverse:
            MatrixPermutation(opt_gate_matrix, np.array([1, 0]), changeInput=True)

        # Overwrite the indexes of the 2-qubits quantum gate in the circuit.
        is_find = False     # find the pre-generate two qubits gate with same qubit indexes
        overwritten_idx = []
        for idx_bit, _ in self._two_qubits_opt_gates_idxes_dict.items():
            if idx_bit == gate_idx_bit:
                is_find = True
                break

            if idx_bit & gate_idx_bit:
                overwritten_idx.append(idx_bit)

        for idx in overwritten_idx:
            del self._two_qubits_opt_gates_idxes_dict[idx]

        # Add the optimized quantum gate
        if is_find:
            opt_gate = self._opt_gates[self._two_qubits_opt_gates_idxes_dict[gate_idx_bit]]
            if is_diagonal and opt_gate.is_diagonal():
                opt_gate.matrix = multiply(opt_gate_matrix, opt_gate.matrix)
            else:
                opt_gate.matrix = dot(opt_gate_matrix, opt_gate.matrix)
        else:
            opt_gate = UnitaryGate()
            opt_gate.name = f"UnitaryGate_{len(self._opt_gates)}"
            opt_gate.targs = two_qubit_gate.targs
            opt_gate.cargs = two_qubit_gate.cargs
            opt_gate.targets = len(two_qubit_gate.targs)
            opt_gate.controls = len(two_qubit_gate.cargs)
            opt_gate.matrix = opt_gate_matrix
            opt_gate.diagonal = is_diagonal

            self._two_qubits_opt_gates_idxes_dict[gate_idx_bit] = len(self._opt_gates)
            self._opt_gates.append(opt_gate)
