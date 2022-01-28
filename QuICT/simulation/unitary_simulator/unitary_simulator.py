import numpy as np
from typing import *

from QuICT.core.gate import BasicGate, CompositeGate
from QuICT.simulation.utils import DisjointSet, dp, build_unitary_gate
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.ops.linalg.gpu_calculator as GPUCalculator

unitary_calculation = CPUCalculator


class UnitarySimulator():
    """
    Algorithms to the unitary matrix of a Circuit.
    """
    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        self._computer = CPUCalculator if device == "CPU" else GPUCalculator
        self._precision = np.complex128 if precision == "double" else np.complex64

    def pretreatment(self, circuit):
        """
        Args:
            circuit(Circuit): the circuit needs pretreatment.

        Return:
            CompositeGate: the gates after pretreatment
        """
        circuit_width = circuit.width()
        gateSet = [np.identity(2, dtype=self._precision) for _ in range(circuit_width)]
        tangle = [i for i in range(circuit.width())]
        gates = CompositeGate()

        for gate in circuit.gates:
            if gate.targets + gate.controls >= 3:
                raise Exception("only support 2-qubit gates and 1-qubit gates.")

            # 1-qubit gate
            if gate.targets + gate.controls == 1:
                target = gate.targ
                if tangle[target] == target:
                    gateSet[target] = self._computer.dot(gate.matrix, gateSet[target])
                else:
                    if tangle[target] < target:
                        gateSet[target] = self._computer.dot(
                            np.kron(np.identity(2, dtype=self._precision), gate.matrix),
                            gateSet[target]
                        )
                    else:
                        gateSet[target] = self._computer.dot(
                            np.kron(gate.matrix, np.identity(2, dtype=self._precision)),
                            gateSet[target]
                        )
                    gateSet[tangle[target]] = gateSet[target]

            # 2-qubit gate
            else:
                affectArgs = gate.cargs + gate.targs
                target1, target2 = affectArgs[0], affectArgs[1]
                if target1 < target2:
                    matrix = gate.matrix
                else:
                    matrix = self._computer.MatrixPermutation(gate.matrix, np.array([1, 0]))

                if tangle[target1] == target2:
                    gateSet[target1] = self._computer.dot(matrix, gateSet[target1])
                    gateSet[target2] = gateSet[target1]
                elif tangle[target1] == target1 and tangle[target2] == target2:
                    if target1 < target2:
                        target_matrix = np.kron(gateSet[target1], gateSet[target2])
                    else:
                        target_matrix = np.kron(gateSet[target2], gateSet[target1])

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
                    gateSet[tangle[target]] = np.identity(2, dtype=self._precision)
                    gateSet[target] = np.identity(2, dtype=self._precision)
                    tangle[tangle[target]] = tangle[target]
                    tangle[target] = target

                    if tangle[revive] == revive:
                        if revive <= target1 and revive <= target2:
                            target_matrix = np.kron(gateSet[revive], np.identity(2, dtype=self._precision))
                        else:
                            target_matrix = np.kron(np.identity(2, dtype=self._precision), gateSet[revive])

                        gateSet[target1] = self._computer.dot(matrix, target_matrix)
                        gateSet[target2] = gateSet[target1]
                        tangle[revive], tangle[target] = target, revive
                    else:
                        build_unitary_gate(gates, gateSet[revive], [revive, tangle[revive]])
                        gateSet[tangle[revive]] = np.identity(2, dtype=self._precision)
                        gateSet[revive] = np.identity(2, dtype=self._precision)
                        tangle[tangle[revive]] = tangle[revive]
                        tangle[revive] = revive

                        gateSet[target1] = matrix
                        gateSet[target2] = gateSet[target1]
                        tangle[revive], tangle[target] = target, revive

        for i in range(circuit_width):
            if tangle[i] == i:
                if not np.allclose(np.identity(2, dtype=self._precision), gateSet[i]):
                    build_unitary_gate(gates, gateSet[i], i)
            elif tangle[i] > i:
                if not np.allclose(np.identity(4, dtype=self._precision), gateSet[i]):
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

    def run(self, circuit, use_previous: bool = False) -> np.ndarray:
        """
        Get the unitary matrix of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The unitary matrix of input circuit.
        """

        qubit = circuit.width()
        if len(circuit.gates) == 0:
            return np.identity(1 << qubit, dtype=self._precision)

        ordering, small_gates = self.unitary_pretreatment(circuit)
        u_mat, u_args = self.merge_unitary_by_ordering(small_gates, ordering)
        result_mat, _ = self.merge_two_unitary(
            np.identity(1 << qubit, dtype=self._precision),
            [i for i in range(qubit)],
            u_mat,
            u_args
        )

        return result_mat

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
            matrix, affectArgs
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
        affectArgs = [0] * len_c
        cnt = len_a
        for rb in range(len_b):
            if args_b[rb] not in seta:
                mps[rb] = cnt
                affectArgs[cnt] = args_b[rb]
                cnt += 1
            else:
                for ra in range(len_a):
                    if args_a[ra] == args_b[rb]:
                        mps[rb] = ra
                        affectArgs[ra] = args_b[rb]
                        break
        cnt = len_b
        for ra in range(len_a):
            if args_a[ra] not in setb:
                mps[cnt] = ra
                affectArgs[ra] = args_a[ra]
                cnt += 1
        mat_b = self._computer.MatrixPermutation(mat_b, np.array(mps))
        res_mat = self._computer.dot(mat_b, mat_a)
        return res_mat, affectArgs

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
            matrix, affectArgs
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
