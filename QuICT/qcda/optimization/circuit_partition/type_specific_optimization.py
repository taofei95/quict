from typing import List, Tuple

from QuICT.core import Circuit
from QuICT.core.utils.circuit_info import CircuitMode
from QuICT.qcda.utility import OutputAligner


class TypeSpecificOptimization:
    """
    Meta optimizer that applies a given circuit optimizer to all sub-circuits of applicable
    gate set in a given circuit.
    """

    def __init__(self, optimizer, circuit_mode: CircuitMode):
        """
        Args:
            optimizer: The original optimizer instance. It needs to have execute() method
            circuit_mode(CircuitMode): The applicable gate set
        """
        self.optimizer = optimizer
        self.circuit_mode = circuit_mode

    def _cut_circuit(self, circuit) -> List[Tuple[Circuit, Circuit]]:
        """
        TODO
        """
        pass

    @OutputAligner
    def execute(self, circuit: Circuit) -> Circuit:
        """
        Optimize a given circuit

        Args:
            circuit(Circuit): the given circuit

        Returns:
            Circuit: the optimized circuit
        """
        ret = Circuit(circuit.width())
        for pred, circ in self._cut_circuit(circuit):
            ret.extend(pred.gates)
            if circ.size():
                new_circ: Circuit = self.optimizer.execute(circ)
                ret.extend(new_circ.gates)
        return ret
