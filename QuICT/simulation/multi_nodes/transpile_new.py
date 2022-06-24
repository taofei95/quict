from imp import source_from_cache
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import DataSwitch, DataSwitchType, DeviceTrigger, Multiply, SpecialGate
from QuICT.qcda.synthesis import GateDecomposition


ALL_PENTY = 0.5
HALF_PENTY = 2
CARG_PENTY = 3
PROB_ADD = 1


NORMAL_GATE_SET_1qubit = [
    GateType.h, GateType.sx, GateType.sy, GateType.sw,
    GateType.u2, GateType.u3, GateType.rx, GateType.ry
]


class Transpile:
    """ Transpile the circuit by the number of devices.

    Args:
        ndev (int): The number of devices.
    """

    def __init__(self, ndev: int):
        self._ndev = ndev
        self._split_qb_num = int(np.log2(ndev))
        assert 2 ** self._split_qb_num == self._ndev

    def _transpile(self, depth_gate: list, intervals: list, split_qubits: list, total_qubits: int):
        print(split_qubits)
        print(intervals)
        t_qubits = total_qubits - self._split_qb_num
        transpiled_circuit = Circuit(t_qubits)
        start_point = 0
        for interval_idx, point in enumerate(intervals):
            current_split_qubit = [split_qubits[interval_idx]]
            for gates in depth_gate[start_point:point]:
                for gate in gates:
                    if gate.type in [GateType.id, GateType.barrier]:
                        continue

                    gate_args = gate.cargs + gate.targs
                    updated_args, union_arg = self._args_adjust(gate_args, current_split_qubit)
                    gate = gate & updated_args
                    if len(union_arg) == 0:
                        if gate.matrix_type == MatrixType.special:
                            SpecialGate(gate.type, gate_args) | transpiled_circuit
                        else:
                            gate | transpiled_circuit

                        continue

                    # Deal with gate has qubits in splited qubits
                    if len(gate_args) == 1:
                        device_trigger = self._single_qubit_gate_transpile(gate, current_split_qubit)
                    elif len(gate_args) == 2:
                        device_trigger = self._double_qubits_gate_transpile(gate, t_qubits, current_split_qubit, union_arg)
                    else:
                        raise ValueError("Only supportted transpilt for gate less than 2 qubits.")

                    device_trigger | transpiled_circuit

            start_point = point
            if interval_idx != len(intervals) - 1 and split_qubits[interval_idx + 1] != current_split_qubit[0]:
                updated_sq_arg, _ = self._args_adjust([split_qubits[interval_idx + 1]], current_split_qubit)
                self._split_qubit_exchange(1, t_qubits - 1 - updated_sq_arg[0]) | transpiled_circuit

        return transpiled_circuit

    def _split_qubit_exchange(self, old_qubit, new_qubit):
        dev_mapping = {}
        for index in range(self._ndev):
            splited_cgate = CompositeGate()
            dest = index ^ old_qubit
            _0_1 = int(index > dest)
            DataSwitch(
                index ^ old_qubit,
                DataSwitchType.ctarg,
                {new_qubit: _0_1}
            ) | splited_cgate

            dev_mapping[index] = splited_cgate

        return DeviceTrigger(dev_mapping)

    def _single_qubit_gate_transpile(
        self,
        gate: BasicGate,
        split_qubits: list
    ):
        gate_arg, matrix_type = gate.targ, gate.matrix_type
        split_idx = 1 << (split_qubits.index(gate_arg))

        dev_mapping = {}
        for index in range(self._ndev):
            _0_1 = index & split_idx
            destination = index ^ split_idx
            splited_cgate = CompositeGate()
            # Normal 1-qubit gates
            if matrix_type == MatrixType.normal:
                DataSwitch(destination, DataSwitchType.half) | splited_cgate
                gate | splited_cgate(0)
                DataSwitch(destination, DataSwitchType.half) | splited_cgate
            # diagonal 1-qubit gates
            elif matrix_type == MatrixType.diagonal:
                value = gate.matrix[_0_1, _0_1]
                Multiply(value) | splited_cgate
            # swap 1-qubit gates [x]
            elif matrix_type == MatrixType.swap:
                DataSwitch(destination, DataSwitchType.all) | splited_cgate
            # reverse 1-qubit gates
            elif matrix_type == MatrixType.reverse:
                value = gate.matrix[int(not _0_1), _0_1]
                Multiply(value) | splited_cgate
                DataSwitch(destination, DataSwitchType.all) | splited_cgate
            # Controlled 1-qubit gates [Z, U1, T, T_dagger, S, S_dagger]
            elif matrix_type == MatrixType.control:
                if _0_1:
                    Multiply(gate.matrix[1, 1]) | splited_cgate
            else:
                SpecialGate(gate.type, gate_arg, split_idx) | splited_cgate

            dev_mapping[index] = splited_cgate

        return DeviceTrigger(dev_mapping)

    def _double_qubits_gate_transpile(
        self,
        gate: BasicGate,
        circuit_qubits: int,
        split_qubits: list,
        union_args: list
    ):
        gate_args, matrix_type = gate.cargs + gate.targs, gate.matrix_type
        double_exceed = False
        if len(union_args) == 2:
            double_exceed = True
            split_qubit_idx = [1 << split_qubits.index(arg) for _, arg in union_args]
        else:
            single_union_idx = union_args[0][0]
            inside_index = single_union_idx ^ 1
            split_qubit_idx = 1 << split_qubits.index(union_args[0][1])

        dev_mapping = {}
        for index in range(self._ndev):
            splited_cgate = CompositeGate()
            # [1, 1, a, b]
            if matrix_type == MatrixType.diagonal:
                if double_exceed:   # both args in splited
                    if index & split_qubit_idx[0]:
                        value = gate.matrix[3, 3] if index & split_qubit_idx[1] else gate.matrix[2, 2]
                        Multiply(value) | splited_cgate
                elif single_union_idx == 0:  # carg in splited
                    if index & split_qubit_idx:
                        Unitary(gate.target_matrix, MatrixType.diagonal) | splited_cgate(gate_args[1])
                else:
                    value = gate.matrix[3, 3] if index & split_qubit_idx else gate.matrix[2, 2]
                    control_matrix = np.array([[1, 0], [0, value]], dtype=np.complex128)
                    Unitary(control_matrix, MatrixType.control) | splited_cgate(gate_args[0])
            # controlled 2-qubits gate [1, 1, 1, a]
            elif matrix_type == MatrixType.control:
                value = gate.matrix[3, 3]
                control_matrix = np.array([[1, 0], [0, value]], dtype=np.complex128)
                if double_exceed:
                    if (index & split_qubit_idx[0]) and (index & split_qubit_idx[1]):
                        Multiply(value) | splited_cgate
                else:
                    if index & split_qubit_idx:
                        Unitary(control_matrix, MatrixType.control) | splited_cgate(gate_args[inside_index])
            # diagonal 2-qubits gate [a, b, c, d]
            elif matrix_type == MatrixType.diag_diag:
                if double_exceed:
                    control_index = 2 if index & split_qubit_idx[0] else 0
                    if index & split_qubit_idx[1]:
                        control_index += 1

                    value = gate.matrix[control_index, control_index]
                    Multiply(value) | splited_cgate
                elif single_union_idx == 0:
                    matrix = gate.matrix[2:, 2:] if index & split_qubit_idx else gate.matrix[:2, :2]
                    Unitary(matrix, MatrixType.diagonal) | splited_cgate(gate_args[1])
                else:
                    _1 = 1 if index & split_qubit_idx else 0
                    matrix = np.identity(2, dtype=np.complex128)
                    matrix[[0, 1], [0, 1]] = gate.matrix[[_1, _1 + 2], [_1, _1 + 2]]
                    Unitary(matrix) | splited_cgate(gate_args[0])
            # control-reverse 2-qubits gate [1, 1, b, a]
            elif matrix_type == MatrixType.reverse:
                if double_exceed:
                    if index & split_qubit_idx[0]:
                        value = gate.matrix[2, 3] if index & split_qubit_idx[1] else gate.matrix[3, 2]
                        Multiply(value) | splited_cgate
                        DataSwitch(index ^ split_qubit_idx[1], DataSwitchType.all) | splited_cgate
                elif single_union_idx == 0:
                    if index & split_qubit_idx:
                        matrix = gate.matrix[2:, 2:]
                        Unitary(matrix, MatrixType.reverse) | splited_cgate(gate_args[1])
                else:
                    matrix = np.identity(2, dtype=np.complex128)
                    matrix[1, 1] = gate.matrix[2, 3] if index & split_qubit_idx else gate.matrix[3, 2]
                    Unitary(matrix) | splited_cgate(gate_args[0])
                    DataSwitch(
                        index ^ split_qubit_idx,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[0]: 1}
                    ) | splited_cgate
            # control-normal 2-qubits gate [CH, CU3]
            elif matrix_type == MatrixType.normal:
                if double_exceed:
                    if index & split_qubit_idx[0]:
                        DataSwitch(index ^ split_qubit_idx[1], DataSwitchType.half) | splited_cgate
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(0)
                        DataSwitch(index ^ split_qubit_idx[1], DataSwitchType.half) | splited_cgate
                elif single_union_idx == 0:
                    if index & split_qubit_idx:
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(gate_args[1])
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[0]: int(index < dest)}
                    ) | splited_cgate
                    if index & split_qubit_idx:
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(gate_args[0])
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[0]: int(index < dest)}
                    ) | splited_cgate
            # Swap gate [swap]
            elif matrix_type == MatrixType.swap:
                if double_exceed:
                    control_index = 2 if index & split_qubit_idx[0] else 0
                    if index & split_qubit_idx[1]:
                        control_index += 1

                    if control_index in [1, 2]:
                        destination = index ^ (sum(split_qubit_idx))
                        DataSwitch(destination, DataSwitchType.all)
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[inside_index]: int(index < dest)}
                    ) | splited_cgate
            # Diagonal * Matrix [Fsim]
            elif matrix_type == MatrixType.ctrl_normal:
                if double_exceed:
                    if index & split_qubit_idx[0] and index & split_qubit_idx[1]:
                        Multiply(gate.matrix[-1, -1]) | splited_cgate
                    elif index & split_qubit_idx[0] or index & split_qubit_idx[1]:
                        destination = index ^ (sum(split_qubit_idx))
                        DataSwitch(destination, DataSwitchType.half) | splited_cgate
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(0)
                        DataSwitch(destination, DataSwitchType.half) | splited_cgate
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[inside_index]: 0}
                    ) | splited_cgate
                    if index & split_qubit_idx:
                        matrix = np.identity(2, dtype=np.complex128)
                        matrix[1, 1] = gate.matrix[3, 3]
                        Unitary(matrix, MatrixType.control) | splited_cgate(gate_args[inside_index])
                    else:
                        Unitary(gate.matrix[1:3, 1:3]) | splited_cgate(gate_args[inside_index])
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[inside_index]: 0}
                    ) | splited_cgate
            # Matrix * Matrix [Rxx, Ryy]
            elif matrix_type == MatrixType.normal_normal:
                outside_matrix = gate.matrix[[0, 0, 3, 3], [0, 3, 0, 3]].reshape(2, 2)
                inside_matrix = gate.matrix[1:3, 1:3]
                if double_exceed:
                    destination = index ^ (sum(split_qubit_idx))
                    DataSwitch(destination, DataSwitchType.half) | splited_cgate
                    matrix = outside_matrix if abs(index - destination) < sum(split_qubit_idx) else inside_matrix
                    Unitary(matrix) | splited_cgate(0)
                    DataSwitch(destination, DataSwitchType.half) | splited_cgate
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[inside_index]: 1}
                    ) | splited_cgate

                    matrix = inside_matrix if index & split_qubit_idx else outside_matrix
                    Unitary(matrix) | splited_cgate(gate_args[inside_index])
                    DataSwitch(
                        dest,
                        DataSwitchType.ctarg,
                        {circuit_qubits - 1 - gate_args[inside_index]: 1}
                    ) | splited_cgate

            dev_mapping[index] = splited_cgate

        return DeviceTrigger(dev_mapping)

    def _args_adjust(self, gate_args, split_args):
        if isinstance(split_args, int):
            split_args = [split_args]

        res = gate_args[:]
        union_arg = []
        for idx, garg in enumerate(gate_args):
            if garg in split_args:
                union_arg.append((idx, garg))

            for sarg in split_args:
                if sarg < garg:
                    res[idx] -= 1

        return res, union_arg

    def _order_gates_by_depth(self, gates: list) -> list:
        gate_by_depth = [[gates[0]]]          # List[list], gates for each depth level.
        gate_args_by_depth = [set(gates[0].cargs + gates[0].targs)]     # List[set], gates' qubits for each depth level.
        for gate in gates[1:]:
            gate_arg = set(gate.cargs + gate.targs)
            for i in range(len(gate_args_by_depth) - 1, -1, -1):
                if gate_arg & gate_args_by_depth[i]:
                    if i == len(gate_args_by_depth) - 1:
                        gate_by_depth.append([gate])
                        gate_args_by_depth.append(gate_arg)
                    else:
                        gate_by_depth[i + 1].append(gate)
                        gate_args_by_depth[i + 1] = gate_arg | gate_args_by_depth[i + 1]
                    break
                else:
                    if i == 0:
                        gate_by_depth[i].append(gate)
                        gate_args_by_depth[i] = gate_arg | gate_args_by_depth[i]

        return gate_by_depth

    def _gate_switch_cost_generator(self, gates_by_depth: list, qubits: int) -> list:
        data_switch_cost_matrix = np.zeros((qubits, len(gates_by_depth)), dtype=np.int32)
        for idx, depth_gates in enumerate(gates_by_depth):
            for gate in depth_gates:
                gate_matrix_type = gate.matrix_type
                gate_args = gate.cargs + gate.targs
                if gate_matrix_type == MatrixType.normal:
                    if gate.controls == 0:
                        data_switch_cost_matrix[gate_args, idx] += HALF_PENTY
                    else:
                        data_switch_cost_matrix[gate.targs, idx] += CARG_PENTY
                elif gate_matrix_type == MatrixType.reverse:
                    if gate.controls == 1:
                        data_switch_cost_matrix[gate.targs, idx] += CARG_PENTY
                elif gate_matrix_type == MatrixType.swap:
                    if gate.targets > 1:
                        data_switch_cost_matrix[gate_args[-1], idx] += CARG_PENTY
                elif gate_matrix_type in [MatrixType.ctrl_normal, MatrixType.normal_normal]:
                    data_switch_cost_matrix[gate_args, idx] += CARG_PENTY
                elif gate.type in [GateType.measure, GateType.reset]:
                    data_switch_cost_matrix[gate_args, idx] += PROB_ADD

        return data_switch_cost_matrix

    def _split_qubits(self, cost_matrix: np.ndarray, cost_threshold: int = 3):
        # divided the cost matrix by longest period with switched cost less than cost_threshold
        qubit_num, gate_depth = cost_matrix.shape
        period_interval = []                # record cost matrix's period interval
        sum_cost_by_longest_period = []     # record the sum of period interval of cost matrix
        period_idx = 0
        while period_idx < gate_depth - 1:
            longest_period_idx = 0
            for qubit in range(qubit_num):
                switch_cost = 0
                for idx in range(period_idx, gate_depth, 1):
                    switch_cost += cost_matrix[qubit, idx]
                    if switch_cost > cost_threshold:
                        break

                if idx > longest_period_idx:
                    longest_period_idx = idx

            switch_cost = np.sum(cost_matrix[:, period_idx:longest_period_idx], axis=1)
            sum_cost_by_longest_period.append(switch_cost)

            period_idx = longest_period_idx
            period_interval.append(period_idx)

        period_interval[-1] += 1
        # find the smallest cost through the sum_cost_by_longest_period
        sum_cost_by_longest_period = np.array(sum_cost_by_longest_period, dtype=np.int32)
        period_qubit_selection = [int(np.argmin(sum_cost_by_longest_period[0]))]
        for sum_cost in sum_cost_by_longest_period[1:]:
            current_minarg = int(np.argmin(sum_cost))
            if current_minarg != period_qubit_selection[-1]:      # required switch split qubits
                # deal with add switch cost threshold
                if sum_cost[period_qubit_selection[-1]] < sum_cost[current_minarg] + cost_threshold:
                    current_minarg = period_qubit_selection[-1]
                # deal with switch back cost
                elif len(period_qubit_selection) >= 2 and period_qubit_selection[-2] == current_minarg:
                    start_idx = len(period_qubit_selection) - 2
                    previous_cost = np.sum(sum_cost_by_longest_period[start_idx:start_idx + 2, current_minarg])
                    current_cost = np.sum(sum_cost_by_longest_period[start_idx:start_idx + 2, period_qubit_selection[-2:]]) + 2 * cost_threshold
                    if previous_cost <= current_cost:
                        period_qubit_selection[-1] = current_minarg

            period_qubit_selection.append(current_minarg)

        return period_interval, period_qubit_selection

    def run(self, circuit: Circuit):
        # step 1: run GateDecomposition to avoid Unitary with large qubits and other special gates
        qubits = circuit.width()
        gates = GateDecomposition.execute(circuit).gates

        # step 2: order the circuit's gates by depth
        depth_gate = self._order_gates_by_depth(gates)

        # step 3: decided split qubits by depth gates
        cost_matrix = self._gate_switch_cost_generator(depth_gate, qubits)
        split_qubit_intervals, selected_qubit_per_intervals = self._split_qubits(cost_matrix)

        # step 4: transpile circuit by split qubits [add op.data_swap, change gate]
        transpiled_circuits = self._transpile(depth_gate, split_qubit_intervals, selected_qubit_per_intervals, qubits)

        return transpiled_circuits, selected_qubit_per_intervals[-1]
