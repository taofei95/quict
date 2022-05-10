import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import DataSwitch, DataSwitchType, DeviceTrigger, Multiply, SpecialGate
from QuICT.qcda.synthesis import GateDecomposition


ALL_PENTY = 0.5
HALF_PENTY = 1
CARG_PENTY = 2
PROB_ADD = 0.5


NORMAL_GATE_SET_1qubit = [
    GateType.h, GateType.sx, GateType.sy, GateType.sw,
    GateType.u2, GateType.u3, GateType.rx, GateType.ry
]


class Transpile:
    def __init__(self, ndev: int):
        self._ndev = ndev
        self._split_qb_num = int(np.log2(ndev))
        assert 2 ** self._split_qb_num == self._ndev

    def _transpile(self, circuit: Circuit, split_qubits: list):
        t_qubits = circuit.width() - self._split_qb_num
        transpiled_circuit = Circuit(t_qubits)
        for gate in circuit.gates:
            if gate.type in [GateType.id, GateType.barrier]:
                continue

            gate_args = gate.cargs + gate.targs
            gate & self._args_adjust(gate_args, split_qubits)
            if len(set(gate_args) & set(split_qubits)) == 0:
                if gate.matrix_type == MatrixType.special:
                    SpecialGate(gate.type, gate_args) | transpiled_circuit
                else:
                    gate | transpiled_circuit

                continue

            # Deal with gate has qubits in splited qubits
            if len(gate_args) == 1:
                device_trigger = self._single_qubit_gate_transpile(gate, split_qubits, t_qubits)
            elif len(gate_args) == 2:
                device_trigger = self._double_qubits_gate_transpile(gate, split_qubits, t_qubits)
            else:
                raise ValueError("Only supportted transpilt for gate less than 2 qubits.")

            device_trigger | transpiled_circuit

        return transpiled_circuit

    def _single_qubit_gate_transpile(
        self,
        gate: BasicGate,
        split_qubits: list,
        max_qubits: int
    ):
        gate_arg, matrix_type = gate.targ, gate.matrix_type
        split_idx = split_qubits.index(gate_arg)

        dev_mapping = {}
        for index in range(self._ndev):
            _0_1 = index & (1 << split_idx)
            destination = index ^ (1 << split_idx)
            splited_cgate = CompositeGate()
            # Normal 1-qubit gates
            if matrix_type == MatrixType.normal:
                DataSwitch(destination, DataSwitchType.half) | splited_cgate
                gate | splited_cgate(max_qubits - 1)
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
                value = gate.matrix[_0_1, int(not _0_1)]
                Multiply(value) | splited_cgate
                DataSwitch(destination, DataSwitchType.all) | splited_cgate
            # Controlled 1-qubit gates [Z, U1, T, T_dagger, S, S_dagger]
            elif matrix_type == MatrixType.control:
                if _0_1:
                    Multiply(gate.matrix[1, 1]) | splited_cgate
            else:
                SpecialGate(gate.type, gate_arg, 1 << split_idx) | splited_cgate

            dev_mapping[index] = splited_cgate

        return DeviceTrigger(dev_mapping)

    def _double_qubits_gate_transpile(
        self,
        gate: BasicGate,
        split_qubits: list,
        max_qubits: int
    ):
        gate_args, matrix_type = gate.cargs + gate.targs, gate.matrix_type
        union_args = set(gate_args) & set(split_qubits)
        double_exceed, outside_index = False, False
        if len(union_args) == 2:
            double_exceed = True
            split_qubit_idx = [1 << split_qubits.index(arg) for arg in gate_args]
        else:
            outside_index = gate_args.index(union_args[0])
            inside_index = outside_index ^ 1
            split_qubit_idx = 1 << split_qubits.index(union_args[0])

        dev_mapping = {}
        for index in range(self._ndev):
            splited_cgate = CompositeGate()
            # [1, 1, a, b]
            if matrix_type == MatrixType.diagonal:
                if double_exceed:   # both args in splited
                    if index & split_qubit_idx[0]:
                        value = gate.matrix[3, 3] if index & split_qubit_idx[1] else gate.matrix[2, 2]
                        Multiply(value) | splited_cgate
                elif outside_index:  # carg in splited
                    if index & split_qubit_idx:
                        Unitary(gate.target_matrix) | splited_cgate(gate_args[0])
                else:
                    value = gate.matrix[3, 3] if index & split_qubit_idx else gate.matrix[2, 2]
                    control_matrix = np.array([[1, 0], [0, value]], dtype=np.complex128)
                    Unitary(control_matrix) | splited_cgate(gate_args[1])
            # controlled 2-qubits gate [1, 1, 1, a]
            elif matrix_type == MatrixType.control:
                value = gate.matrix[3, 3]
                control_matrix = np.array([[1, 0], [0, value]], dtype=np.complex128)
                if double_exceed:
                    if (index & split_qubit_idx[0]) and (index & split_qubit_idx[1]):
                        Multiply(value) | splited_cgate
                else:
                    if index & split_qubit_idx:
                        Unitary(control_matrix) | splited_cgate(gate_args[inside_index])
            # diagonal 2-qubits gate [a, b, c, d]
            elif matrix_type == MatrixType.diag_diag:
                if double_exceed:
                    control_index = 2 if index & split_qubit_idx[0] else 0
                    if index & split_qubit_idx[1]:
                        control_index += 1

                    value = gate.matrix[control_index, control_index]
                    Multiply(value) | splited_cgate
                elif outside_index:
                    matrix = gate.matrix[2:, 2:] if index & split_qubit_idx else gate.matrix[:2, :2]
                    Unitary(matrix) | splited_cgate(gate_args[1])
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
                elif outside_index:
                    if index & split_qubit_idx:
                        matrix = gate.matrix[2:, 2:]
                        Unitary(matrix) | splited_cgate(gate_args[1])
                else:
                    matrix = np.identity(2, dtype=np.complex128)
                    matrix[1, 1] = gate.matrix[2, 3] if index & split_qubit_idx else gate.matrix[3, 2]
                    Unitary(matrix) | splited_cgate(gate_args[0])
                    DataSwitch(index ^ split_qubit_idx, DataSwitchType.ctarg, {gate_args[0]: 1}) | splited_cgate
            # control-normal 2-qubits gate [CH, CU3]
            elif matrix_type == MatrixType.normal:
                if double_exceed:
                    if index & split_qubit_idx[0]:
                        DataSwitch(index ^ split_qubit_idx[1], DataSwitchType.half) | splited_cgate
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(max_qubits - 1)
                        DataSwitch(index ^ split_qubit_idx[1], DataSwitchType.half) | splited_cgate
                elif outside_index:
                    if index & split_qubit_idx:
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(gate_args[1])
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[0]: int(index < dest)}) | splited_cgate
                    if index & split_qubit_idx:
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(gate_args[0])
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[0]: int(index < dest)}) | splited_cgate
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
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[inside_index]: int(index < dest)}) | splited_cgate
            # Diagonal * Matrix [Fsim]
            elif matrix_type == MatrixType.ctrl_normal:
                if double_exceed:
                    if index & split_qubit_idx[0] and index & split_qubit_idx[1]:
                        Multiply(gate.matrix[-1, -1]) | splited_cgate
                    elif index & split_qubit_idx[0] or index & split_qubit_idx[1]:
                        destination = index ^ (sum(split_qubit_idx))
                        DataSwitch(destination, DataSwitchType.half) | splited_cgate
                        Unitary(gate.matrix[2:, 2:]) | splited_cgate(max_qubits - 1)
                        DataSwitch(destination, DataSwitchType.half) | splited_cgate
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[inside_index]: 0}) | splited_cgate
                    if index & split_qubit_idx:
                        matrix = np.identity(2, dtype=np.complex128)
                        matrix[1, 1] = gate.matrix[3, 3]
                        Unitary(matrix) | splited_cgate(gate_args[inside_index])
                    else:
                        Unitary(gate.matrix[1:3, 1:3]) | splited_cgate(gate_args[inside_index])
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[inside_index]: 0}) | splited_cgate
            # Matrix * Matrix [Rxx, Ryy]
            elif matrix_type == MatrixType.normal:
                outside_matrix = gate.matrix[[0, 0, 3, 3], [0, 3, 0, 3]].reshape(2, 2)
                inside_matrix = gate.matrix[1:3, 1:3]
                if double_exceed:
                    destination = index ^ (sum(split_qubit_idx))
                    DataSwitch(destination, DataSwitchType.half) | splited_cgate
                    matrix = outside_matrix if abs(index - destination) < sum(split_qubit_idx) else inside_matrix
                    Unitary(matrix) | splited_cgate(max_qubits - 1)
                    DataSwitch(destination, DataSwitchType.half) | splited_cgate
                else:
                    dest = index ^ split_qubit_idx
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[inside_index]: 0}) | splited_cgate

                    matrix = outside_matrix if index & split_qubit_idx else inside_matrix
                    Unitary(matrix) | splited_cgate(max_qubits - 1)
                    DataSwitch(dest, DataSwitchType.ctarg, {gate_args[inside_index]: 0}) | splited_cgate

            dev_mapping[index] = splited_cgate

        return dev_mapping

    def _args_adjust(self, gate_args, split_args):
        res = gate_args[:]
        for idx, garg in enumerate(gate_args):
            if garg in split_args:
                continue

            for sarg in split_args:
                if sarg < garg:
                    res[idx] -= 1

        return res

    def _split_qubits(self, circuit: Circuit) -> list:
        qubits = circuit.width()
        comm_cost = np.array([0] * qubits, dtype=np.float32)
        for gate in circuit.gates:
            # Consider trigger here
            gate_type = gate.type
            gate_args = gate.cargs + gate.targs
            if gate_type in NORMAL_GATE_SET_1qubit:
                comm_cost[gate_args[0]] += HALF_PENTY
            elif gate_type in [GateType.x, GateType.y]:
                comm_cost[gate_args[0]] += ALL_PENTY
            elif gate_type in [
                GateType.cx, GateType.cy, GateType.ch, GateType.cu3,
                GateType.fsim, GateType.Rxx, GateType.Ryy, GateType.swap,
                GateType.unitary
            ]:
                for arg in gate_args:
                    comm_cost[arg] += CARG_PENTY
            elif gate_type in [GateType.measure, GateType.reset]:
                comm_cost[arg] += PROB_ADD
            else:
                continue

        split_qubits = []
        for _ in range(self._split_qb_num):
            min_idx = np.argmin(comm_cost)
            split_qubits.append(min_idx)
            comm_cost[min_idx] = np.Infinity

        return split_qubits

    def run(self, circuit: Circuit):
        # step 1: run gatedecomposition
        circuit.gates = GateDecomposition.execute(circuit).gates

        # step 2: decided split qubits
        split_qubits = self._split_qubits(circuit)

        # step 3: transpile circuit by split qubits [add op.data_swap, change gate]
        transpiled_circuits = self._transpile(circuit, split_qubits)

        return transpiled_circuits
