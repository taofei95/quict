import numpy as np

from ..disjoint_set import DisjointSet

from .._simulation import BasicSimulator

from QuICT.core import *
from QuICT.ops.linalg.unitary_calculation import *


class UnitarySimulator(BasicSimulator):
    """
    Algorithms to the unitary matrix of a Circuit.
    """

    @classmethod
    def run(cls, circuit: Circuit) -> np.ndarray:
        """
        Get the unitary matrix of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The unitary matrix of input circuit.
        """

        qubit = circuit.circuit_width()
        if len(circuit.gates) == 0:
            return np.identity(1 << qubit, dtype=np.complex128)
        ordering, small_gates = BasicSimulator.unitary_pretreatment(circuit)
        print(ordering)
        u_gate = cls.merge_unitary_by_ordering(small_gates, ordering)
        unitary = Unitary(np.identity(1 << qubit, dtype=np.complex128)) & [i for i in range(qubit)]
        u_gate = UnitarySimulator.merge_two_unitary(unitary, u_gate)

        return u_gate.matrix

    @classmethod
    def merge_two_unitary(cls, gate_a: BasicGate, gate_b: BasicGate) -> UnitaryGate:
        """
        Combine 2 gates into a new unitary gate.

        Args:
            gate_a (BasicGate): Gate in the left.
            gate_b (BasicGate): Gate in the right.

        Returns:
            UnitaryGate: Combined gate with matrix and affectArgs set properly.
        """

        args_a = gate_a.affectArgs
        args_b = gate_b.affectArgs

        seta = set(args_a)
        setb = set(args_b)

        if len(seta & setb) == 0:
            args_b.extend(args_a)
            return Unitary(tensor(gate_b.compute_matrix, gate_a.compute_matrix)) & args_b

        setc = seta | setb
        len_a = len(seta)
        len_b = len(setb)
        len_c = len(setc)

        if len_c == len_a:
            mat_a = gate_a.compute_matrix
        else:
            mat_a = MatrixTensorI(gate_a.compute_matrix, 1, 1 << (len_c - len_a))
        if len_c == len_b:
            mat_b = gate_b.compute_matrix
        else:
            mat_b = MatrixTensorI(gate_b.compute_matrix, 1, 1 << (len_c - len_b))

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
        mat_b = MatrixPermutation(mat_b, np.array(mps))
        gate = Unitary(dot(mat_b, mat_a)) & affectArgs
        return gate

    @classmethod
    def merge_unitary_by_ordering(cls, gates: List[BasicGate], ordering: List[int]) -> BasicGate:
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
            BasicGate: Combined large gate
        """

        print(ordering)
        len_gate = len(gates)

        dSet = DisjointSet(len_gate)
        if len(ordering) + 1 != len(gates):
            assert 0
        x = 0
        for order in ordering:
            order_left = dSet.find(order)
            order_right = dSet.find(order + 1)
            gateA = gates[order_left]
            gateB = gates[order_right]
            x = dSet.union(order_left, order_right)
            gates[x] = UnitarySimulator.merge_two_unitary(gateA, gateB)
        return gates[x]
