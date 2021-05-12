import numpy as np
from QuICT.core import *
from QuICT.ops.linalg.unitary_calculation import *


class UnitarySimulator:
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

        # TODO: Get proper ordering.
        ordering = []

        u_gate = cls.merge_unitary_by_ordering(circuit.gates, ordering)

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

        set1 = set(gate_a.affectArgs)
        set2 = set(gate_b.affectArgs)
        set0 = set1 | set2

        if (len(set1 & set2) == 0):
            mps = list(set0)
            return Unitary(tensor(gate_a.compute_matrix, gate_b.compute_matrix)) & mps

        if len(set0) == len(set1):
            mat_a = gate_a.compute_matrix
        else:
            mat_a = MatrixTensorI(gate_a.compute_matrix, 1, len(set0) - len(set1) + 1)
        if len(set0) == len(set2):
            mat_b = gate_b.compute_matrix
        else:
            mat_b = MatrixTensorI(gate_b.compute_matrix, 1, len(set0) - len(set2) + 1)

        mps = [-1] * len(set0)
        cnt = len(set1)
        mark = [0] * len(set1)
        for rb in range(len(set2)):
            if gate_b.affectArgs[rb] not in set1:
                if (cnt < len(set0)):
                    mps[rb] = cnt
                    cnt += 1
                else:
                    for ra in range(len(set1)):
                        if mark[ra] == 0:
                            mps[rb] = ra
                            mark[ra] = 1
                            break
            else:
                for ra in range(len(set1)):
                    if (gate_a.affectArgs[ra] == gate_b.affectArgs[rb]):
                        mps[rb] = ra
                        mark[ra] = 1
                        break
        for rb in range(len(set2), len(set0)):
            if (cnt < len(set0)):
                mps[rb] = cnt
                cnt += 1
            else:
                for ra in range(len(set1)):
                    if mark[ra] == 0:
                        mps[rb] = ra
                        mark[ra] = 1
                        break

        mat_b = MatrixPermutation(mat_b, mps)
        for r in range(len(mps)):
            mps[r] += min(gate_a.affectArgs[0], gate_b.affectArgs[0])

        gate = Unitary(multiply(mat_a, mat_b)) & mps
        return gate

    @classmethod
    def merge_unitary_by_ordering(cls, gates: List[BasicGate], ordering: List[int]) -> UnitaryGate:
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
            UnitaryGate: Combined large gate.
        """

        # TODO: Implementation.
        """
        （之前写的错误代码）
        midGate = UnitarySimulator.merge_two_unitary(gates[ordering[0]], gates[ordering[1]])
        for r in range(2, ordering.__len__()):
            midGate = UnitarySimulator.merge_two_unitary(midGate, gates[ordering[r]])
        return midGate
        """
        pass
