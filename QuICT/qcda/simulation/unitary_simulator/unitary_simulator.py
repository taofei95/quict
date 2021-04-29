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

        # TODO: Implementation.

        pass

    @classmethod
    def merge_unitary_by_ordering(cls, gates: List[BasicGate], ordering: List[int]) -> UnitaryGate:
        """
        Merge a gate sequence into single unitary gate. The combination order is determined by
        input parameter.

        Args:
            gates (List[BasicGate]): A list consisting of n gates to be merged.
            ordering (List[int]): A permutation of [0,n-1) denoting the combination order of gates.
                If number i is at the j-th position, i-th merge operation would combine 2 gates
                around j-th seam (Remember that those 2 gates might have already been merged into larger
                gates).

        Returns:
            UnitaryGate: Combined large gate.
        """

        # TODO: Implementation.

        pass
