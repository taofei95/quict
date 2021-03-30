from typing import *

from .._synthesis import Synthesis
from QuICT.core import *

class MappingBuilder(Synthesis):
    @staticmethod
    def remap(
            qubit_num: int,
            gates: Sequence[BasicGate],
            mapping: Sequence[int] = None,
    ) -> None:
        """
        Build gates with given mapping via an external gate_builder.

        Args:
            qubit_num (int): Total number of qubit.
            gates (Sequence[BasicGate]): Gates to be remapped.
            It should return gates built without mapping.
            mapping (Sequence[int]): Qubit ordering.

        Returns:
            Sequence[BasicGate]: Synthesized gates with given mapping.
        """
        if mapping is None:
            mapping = [i for i in range(qubit_num)]
        for gate in gates:
            for idx, val in enumerate(gate.cargs):
                gate.cargs[idx] = mapping[val]
            for idx, val in enumerate(gate.targs):
                gate.targs[idx] = mapping[val]