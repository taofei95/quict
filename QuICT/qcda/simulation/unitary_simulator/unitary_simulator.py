import numpy as np
import numba as nb
from typing import *

from ..disjoint_set import DisjointSet

from .._simulation import BasicSimulator

from QuICT.core import *
# import QuICT.ops.linalg.unitary_calculation as unitary_calculation
from QuICT.ops.linalg.cpu_calculator import CPUCalculator

unitary_calculation = CPUCalculator

# from QuICT.ops.linalg.unitary_calculation import *

# TODO: Check platform features by QuICT.utility.xxx
GPU_AVAILABLE = False
GPU_OUT = False


class UnitarySimulator(BasicSimulator):
    """
    Algorithms to the unitary matrix of a Circuit.
    """

    @staticmethod
    def run(circuit: Circuit) -> np.ndarray:
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
        u_mat, u_args = UnitarySimulator.merge_unitary_by_ordering(small_gates, ordering)
        result_mat, _ = UnitarySimulator.merge_two_unitary(
            np.identity(1 << qubit, dtype=np.complex128),
            [i for i in range(qubit)],
            u_mat,
            u_args
        )

        return result_mat.get() if GPU_AVAILABLE else result_mat

    @staticmethod
    def merge_two_unitary(
            mat_a_: np.ndarray,
            args_a: List[int],
            mat_b_: np.ndarray,
            args_b: List[int],
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
            return unitary_calculation.tensor(mat_b_, mat_a_), args_b

        setc = seta | setb
        len_a = len(seta)
        len_b = len(setb)
        len_c = len(setc)

        if len_c == len_a:
            mat_a = mat_a_
        else:
            mat_a = unitary_calculation.MatrixTensorI(mat_a_, 1, 1 << (len_c - len_a))
        if len_c == len_b:
            mat_b = mat_b_
        else:
            mat_b = unitary_calculation.MatrixTensorI(mat_b_, 1, 1 << (len_c - len_b))

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
        mat_b = unitary_calculation.MatrixPermutation(mat_b, np.array(mps))
        res_mat = unitary_calculation.dot(mat_b, mat_a)
        return res_mat, affectArgs

    @staticmethod
    def merge_unitary_by_ordering(
            gates: List[BasicGate],
            ordering: List[int],
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

        print(ordering)
        len_gate = len(gates)

        d_set = DisjointSet(len_gate)
        if len(ordering) + 1 != len(gates):
            raise IndexError("Length not match!")
        matrices = [gates[i].matrix for i in range(len_gate)]
        mat_args = [gates[i].affectArgs for i in range(len_gate)]
        x = 0
        for order in ordering:
            order_left = d_set.find(order)
            order_right = d_set.find(order + 1)
            x = d_set.union(order_left, order_right)
            matrices[x], mat_args[x] = UnitarySimulator.merge_two_unitary(
                matrices[order_left],
                mat_args[order_left],
                matrices[order_right],
                mat_args[order_right],
            )
        res_mat = matrices[x]
        res_arg = mat_args[x]
        return res_mat, res_arg
