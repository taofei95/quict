from typing import *
import numpy as np
from QuICT.core.circuit.circuit import Circuit

from QuICT.core.gate import BasicGate, CompositeGate, GateType
from QuICT.core.utils.gate_type import MatrixType
from QuICT.simulation.utils import DisjointSet, dp, build_unitary_gate
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
from QuICT.simulation.optimization import Optimizer


class UnitarySimulator():
    """ Algorithms to calculate the unitary matrix of a quantum circuit, and simulate.

    Args:
        device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
        precision (str, optional): The precision for the unitary matrix, one of [single, double]. Defaults to "double".
    """

    def __init__(
            self,
            device: str = "CPU",
            precision: str = "double"
    ):
        self._device = device
        self._precision = np.complex128 if precision == "double" else np.complex64
        self._vector = None
        self._circuit = None

        if device == "CPU":
            self._computer = CPUCalculator
            self._array_helper = np
        else:
            import cupy as cp
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator

            self._computer = GPUCalculator
            self._array_helper = cp

    def pretreatment(self, circuit):
        """
        Args:
            circuit(Circuit): the circuit needs pretreatment.

        Return:
            CompositeGate: the gates after pretreatment
        """
        qubits_num = circuit.width()
        gateSet = [self._array_helper.identity(2, dtype=self._precision) for _ in range(qubits_num)]
        tangle = [i for i in range(qubits_num)]
        gates = CompositeGate()
        for gate in circuit.gates:
            if gate.type == GateType.measure:
                continue

            if gate.targets + gate.controls >= 3:
                raise Exception("only support 2-qubit gates and 1-qubit gates.")

            gate_matrix = self._array_helper.array(gate.matrix) if self._device == "GPU" else gate.matrix
            gate_args = gate.cargs + gate.targs
            # 1-qubit gate
            if gate.targets + gate.controls == 1:
                target = gate_args[0]
                if tangle[target] == target:
                    gateSet[target] = self._computer.dot(gate_matrix, gateSet[target])
                else:
                    if tangle[target] < target:
                        gateSet[target] = self._computer.dot(
                            self._array_helper.kron(self._array_helper.identity(2, dtype=self._precision), gate_matrix),
                            gateSet[target]
                        )
                    else:
                        gateSet[target] = self._computer.dot(
                            self._array_helper.kron(gate_matrix, self._array_helper.identity(2, dtype=self._precision)),
                            gateSet[target]
                        )
                    gateSet[tangle[target]] = gateSet[target]

            # 2-qubit gate
            else:
                target1, target2 = gate_args[0], gate_args[1]
                if target1 < target2:
                    matrix = gate_matrix
                else:
                    matrix = self._computer.MatrixPermutation(gate_matrix, self._array_helper.array([1, 0]))

                if tangle[target1] == target2:
                    gateSet[target1] = self._computer.dot(matrix, gateSet[target1])
                    gateSet[target2] = gateSet[target1]
                elif tangle[target1] == target1 and tangle[target2] == target2:
                    if target1 < target2:
                        target_matrix = self._array_helper.kron(gateSet[target1], gateSet[target2])
                    else:
                        target_matrix = self._array_helper.kron(gateSet[target2], gateSet[target1])

                    gateSet[target1] = self._computer.dot(matrix, target_matrix)
                    gateSet[target2] = gateSet[target1]
                    tangle[target1], tangle[target2] = target2, target1
                else:
                    if tangle[target1] != target1:
                        revive = target2
                        target = target1
                    else:
                        revive = target1
                        target = target2

                    build_unitary_gate(gates, gateSet[target], [target, tangle[target]])
                    gateSet[tangle[target]] = self._array_helper.identity(2, dtype=self._precision)
                    gateSet[target] = self._array_helper.identity(2, dtype=self._precision)
                    tangle[tangle[target]] = tangle[target]
                    tangle[target] = target

                    if tangle[revive] == revive:
                        if revive <= target1 and revive <= target2:
                            target_matrix = self._array_helper.kron(gateSet[revive], self._array_helper.identity(2, dtype=self._precision))
                        else:
                            target_matrix = self._array_helper.kron(self._array_helper.identity(2, dtype=self._precision), gateSet[revive])

                        gateSet[target1] = self._computer.dot(matrix, target_matrix)
                        gateSet[target2] = gateSet[target1]
                        tangle[revive], tangle[target] = target, revive
                    else:
                        build_unitary_gate(gates, gateSet[revive], [revive, tangle[revive]])
                        gateSet[tangle[revive]] = self._array_helper.identity(2, dtype=self._precision)
                        gateSet[revive] = self._array_helper.identity(2, dtype=self._precision)
                        tangle[tangle[revive]] = tangle[revive]
                        tangle[revive] = revive

                        gateSet[target1] = matrix
                        gateSet[target2] = gateSet[target1]
                        tangle[revive], tangle[target] = target, revive

        for i in range(qubits_num):
            if tangle[i] == i:
                if not self._array_helper.allclose(
                    self._array_helper.identity(2, dtype=self._precision),
                    gateSet[i]
                ):
                    build_unitary_gate(gates, gateSet[i], i)
            elif tangle[i] > i:
                if not self._array_helper.allclose(
                    self._array_helper.identity(4, dtype=self._precision),
                    gateSet[i]
                ):
                    build_unitary_gate(gates, gateSet[i], [i, tangle[i]])

        return gates

    def unitary_pretreatment(self, circuit):
        small_gates = self.pretreatment(circuit)
        gates = []
        for gate in small_gates.gates:
            gates.append(gate.cargs[:] + gate.targs[:])

        # gates as input
        _, pre = self.unitary_merge_layer(gates)
        order = []

        def pre_search(left, right):
            if left >= right:
                return

            stick = pre[left][right]
            order.append(stick)
            pre_search(left, stick)
            pre_search(stick + 1, right)

        pre_search(0, len(gates) - 1)
        order.reverse()
        return order, small_gates

    def vector_pretreatment(self, circuit):
        small_gates = self.pretreatment(circuit)
        gates = []
        for gate in small_gates.gates:
            gates.append(gate.cargs[:] + gate.targs[:])

        # gates as input
        f, pre = self.unitary_merge_layer(gates)
        gate_length = len(gates)
        width = circuit.width()

        amplitude_f = []
        pre_amplitude = []
        for i in range(gate_length):
            pre_temp = 0
            pre_value = f[0][i].amplitude_cost(width)
            for j in range(i):
                new_value = amplitude_f[j] + f[j + 1][i].amplitude_cost(width)
                if new_value < pre_value:
                    pre_value = new_value
                    pre_temp = j

            amplitude_f.append(pre_value)
            pre_amplitude.append(pre_temp)

        order = []

        def pre_search(left, right):
            if left >= right:
                return

            stick = pre[left][right]
            order.append(stick)
            pre_search(left, stick)
            pre_search(stick + 1, right)

        def pre_amplitude_search(right):
            stick = pre_amplitude[right]
            order.append(-(stick + 1))
            pre_search(stick, right)
            if stick <= 0:
                return

            pre_amplitude_search(stick)

        pre_amplitude_search(gate_length - 1)
        order.reverse()
        return order, small_gates

    def unitary_merge_layer(self, gates: list):
        gate_length = len(gates)
        f = [[None if j != i else dp(gates[i]) for j in range(gate_length)] for i in range(gate_length)]
        pre = [[0 for _ in range(gate_length)] for _ in range(gate_length)]

        for interval in range(1, gate_length):
            for j in range(gate_length - interval):
                pre_temp = j
                pre_value = f[j][j].merge_value(f[j + 1][j + interval])
                for k in range(j + 1, j + interval - 1):
                    new_value = f[j][k].merge_value(f[k + 1][j + interval])
                    if new_value < pre_value:
                        pre_value = new_value
                        pre_temp = k

                f[j][j + interval] = f[j][pre_temp].merge(f[pre_temp + 1][j + interval], pre_value)
                pre[j][j + interval] = pre_temp

        return f, pre

    def merge_two_unitary(
            self,
            mat_a_: np.ndarray,
            args_a: List[int],
            mat_b_: np.ndarray,
            args_b: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Combine 2 gates into a new unitary gate.

        Returns:
            matrix, gate_args
        """

        seta = set(args_a)
        setb = set(args_b)

        if len(seta & setb) == 0:
            args_b.extend(args_a)
            return self._computer.tensor(mat_b_, mat_a_), args_b

        setc = seta | setb
        len_a = len(seta)
        len_b = len(setb)
        len_c = len(setc)

        if len_c == len_a:
            mat_a = mat_a_
        else:
            mat_a = self._computer.MatrixTensorI(mat_a_, 1, 1 << (len_c - len_a))
        if len_c == len_b:
            mat_b = mat_b_
        else:
            mat_b = self._computer.MatrixTensorI(mat_b_, 1, 1 << (len_c - len_b))

        mps = [0] * len_c
        gate_args = [0] * len_c
        cnt = len_a
        for rb in range(len_b):
            if args_b[rb] not in seta:
                mps[rb] = cnt
                gate_args[cnt] = args_b[rb]
                cnt += 1
            else:
                for ra in range(len_a):
                    if args_a[ra] == args_b[rb]:
                        mps[rb] = ra
                        gate_args[ra] = args_b[rb]
                        break
        cnt = len_b
        for ra in range(len_a):
            if args_a[ra] not in setb:
                mps[cnt] = ra
                gate_args[ra] = args_a[ra]
                cnt += 1
        mat_b = self._computer.MatrixPermutation(mat_b, self._array_helper.array(mps))
        res_mat = self._computer.dot(mat_b, mat_a)
        return res_mat, gate_args

    def merge_unitary_by_ordering(
            self,
            gates: List[BasicGate],
            ordering: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Merge a gate sequence into single unitary gate. The combination order is determined by
        input parameter.

        Args:
            gates (List[BasicGate]): A list consisting of n gates to be merged.
            ordering (List[int]): A permutation of [0,n-1] denoting the combination order of gates.
                If number i is at the j-th position, i-th merge operation would combine 2 gates
                around j-th seam (Remember that those 2 gates might have already been merged into larger
                gates).

        Returns:
            matrix, gate_args
        """
        len_gate = gates.size()
        d_set = DisjointSet(len_gate)
        if len(ordering) + 1 != len_gate:
            raise IndexError("Length not match!")

        matrices = [gates[i].matrix for i in range(len_gate)]
        mat_args = [gates[i].cargs + gates[i].targs for i in range(len_gate)]
        x = 0
        for order in ordering:
            order_left = d_set.find(order)
            order_right = d_set.find(order + 1)
            x = d_set.union(order_left, order_right)
            matrices[x], mat_args[x] = self.merge_two_unitary(
                matrices[order_left],
                mat_args[order_left],
                matrices[order_right],
                mat_args[order_right],
            )

        res_mat = matrices[x]
        res_arg = mat_args[x]
        return res_mat, res_arg

    def get_unitary_matrix_new(self, circuit: Circuit = None):
        if circuit is not None:
            self.initial_circuit(circuit)

        assert self._circuit is not None
        gates_order_by_depth = self._circuit.get_gates_order_by_depth()
        unitary_per_qubits = []     # List[List[unitary_matrix(np/cp.ndarray), target_qubits(List[int])]]
        inside_qubits = {}          # Dict[qubit_idx: related index in unitary_per_qubits]
        # TODO: add GroupGateSet for related 2-qubits gates
        for gates in gates_order_by_depth:
            for gate in gates:
                if gate.controls + gate.targets >= 3:
                    raise Exception("only support 2-qubit gates and 1-qubit gates.")

                if gate.matrix_type == MatrixType.special:
                    continue

                args = gate.cargs + gate.targs
                matrix = gate.matrix if self._device == "CPU" else self._array_helper.array(gate.matrix)
                if len(args) == 1:      # Deal with single-qubit gate
                    if args[0] in inside_qubits.keys():
                        related_unitary, related_unitary_args = unitary_per_qubits[inside_qubits[args[0]]]
                        merged_unitary, merged_unitary_args = self.merge_unitaries(
                            related_unitary, related_unitary_args,
                            matrix, args
                        )
                        unitary_per_qubits[inside_qubits[args[0]]] = [merged_unitary, merged_unitary_args]
                    else:
                        unitary_per_qubits.append([matrix, args])
                        inside_qubits[args[0]] = len(unitary_per_qubits) - 1
                else:       # Deal with double-qubits gate
                    if args[0] > args[1]:
                        self._computer.MatrixPermutation(matrix, self._array_helper.array([1, 0]), changeInput=True)
                        args.sort()

                    intersect_qubits = set(args) & set(inside_qubits.keys())
                    if len(intersect_qubits) == 2:
                        related_idx0 = inside_qubits[args[0]]
                        related_idx1 = inside_qubits[args[1]]
                        if related_idx0 == related_idx1:
                            related_unitary, related_unitary_args = unitary_per_qubits[related_idx0]
                            merged_unitary, merged_unitary_args = self.merge_unitaries(
                                related_unitary, related_unitary_args,
                                matrix, args
                            )
                            unitary_per_qubits[related_idx0] = [merged_unitary, merged_unitary_args]
                            continue

                    unitary_per_qubits.append([matrix, args])
                    inside_qubits[args[0]] = len(unitary_per_qubits) - 1
                    inside_qubits[args[1]] = len(unitary_per_qubits) - 1

        half_size = len(unitary_per_qubits) // 2
        left_unitary, left_unitary_args = self._combined_unitaries_by_qubits(unitary_per_qubits[:half_size])
        right_unitary, right_unitary_args = self._combined_unitaries_by_qubits(unitary_per_qubits[half_size:])
        based_unitary, based_unitary_args = self.merge_unitaries(
            left_unitary, left_unitary_args,
            right_unitary, right_unitary_args
        )
        if len(based_unitary_args) != self._qubits_num:
            return self._tensor_unitary(based_unitary, based_unitary_args, list(range(self._qubits_num)))

        return self._computer.MatrixPermutation(based_unitary, np.array(based_unitary_args))

    def merge_unitaries(self, u1, u1_args, u2, u2_args):
        u1_args_set, u2_args_set = set(u1_args), set(u2_args)
        insection_args = u1_args_set & u2_args_set
        if len(insection_args) == 0:
            return self._computer.tensor(u1, u2), u1_args + u2_args

        if u1_args_set == u2_args_set:
            args_idx = [u1_args.index(u2_arg) for u2_arg in u2_args]
            self._computer.MatrixPermutation(
                u2,
                self._array_helper.array(args_idx),
                changeInput=True
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
                self._computer.MatrixPermutation(
                    unitary,
                    self._array_helper.arange(len(unitary_args) - 1, -1, -1),
                    changeInput=True
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

    def _combined_unitaries_by_qubits(self, unitaries):
        idx = np.arange(len(unitaries), dtype=np.int)

        based_unitary, based_unitary_args = unitaries[idx[0]]
        for i in idx[1:]:
            ut, ut_args = unitaries[i]
            based_unitary, based_unitary_args = self.merge_unitaries(
                based_unitary, based_unitary_args,
                ut, ut_args
            )

        return based_unitary, based_unitary_args

    def run(self, circuit, use_previous: bool = False) -> np.ndarray:
        import time

        stime = time.time()
        self.initial_circuit(circuit)
        if not use_previous or self._vector is not None:
            self.initial_vector_state()

        if len(circuit.gates) == 0:
            return self._vector

        et_init = time.time()
        print(f"vector/circuit initial time: {et_init - stime}")
        # Step 1: Generate the unitary matrix of the given circuit
        unitary_matrix = self.get_unitary_matrix()
        et_unitary = time.time()
        print(f"generate unitary matrix time: {et_unitary - et_init}")
        # Step 2: Simulation with the unitary matrix and qubit's state vector
        self._run(unitary_matrix)
        et_simulate = time.time()
        print(f"simulation time: {et_simulate - et_unitary}")
        # Step 3: return the state vector after simulation
        return self._vector

    def initial_circuit(self, circuit):
        self._qubits_num = circuit.width()
        self._circuit = circuit

    def initial_vector_state(self):
        """ Initial the state vector for simulation through UnitarySimulator,
        must after initial_circuit
        """
        self._vector = self._array_helper.zeros(1 << self._qubits_num, dtype=self._precision)
        if self._device == "CPU":
            self._vector[0] = self._precision(1)
        else:
            self._vector.put(0, self._precision(1))

    def get_unitary_matrix(self, circuit: Circuit = None) -> np.ndarray:
        """
        Get the unitary matrix of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated, if None.

        Returns:
            np.ndarray: The unitary matrix of input circuit.
        """
        if circuit is not None:
            self.initial_circuit(circuit)

        assert self._circuit is not None
        ordering, small_gates = self.unitary_pretreatment(self._circuit)
        u_mat, u_args = self.merge_unitary_by_ordering(small_gates, ordering)
        result_mat, _ = self.merge_two_unitary(
            self._array_helper.identity(1 << self._qubits_num, dtype=self._precision),
            [i for i in range(self._qubits_num)],
            u_mat,
            u_args
        )

        return result_mat

    def _run(self, matrix):
        if self._device == "CPU":
            self._vector = self._computer.dot(matrix, self._vector)
        else:
            matrix = self._array_helper.array(matrix)
            self._vector = self._computer.dot(
                matrix,
                self._vector
            )

    def sample(self, shots: int):
        """_summary_

        Args:
            shots (int): _description_

        Returns:
            _type_: _description_
        """
        assert self._circuit is not None
        original_sv = self._vector.copy()
        counts = [0] * (1 << self._qubits_num)
        for _ in range(shots):
            for i in range(self._qubits_num):
                self._measure(i)

            counts[int(self._circuit.qubits)] += 1
            self._vector = original_sv.copy()

        return counts

    def _measure(self, index):
        if self._device == "CPU":
            result = self._computer.measure_gate_apply(
                index,
                self._vector
            )
        else:
            from QuICT.ops.gate_kernel import apply_measuregate, measured_prob_calculate
            prob = measured_prob_calculate(
                index,
                self._vector,
                self._qubits_num
            )
            result = apply_measuregate(
                index,
                self._vector,
                self._qubits_num,
                prob=prob
            )

        self._circuit.qubits[self._qubits_num - 1 - index].measured = int(result)
