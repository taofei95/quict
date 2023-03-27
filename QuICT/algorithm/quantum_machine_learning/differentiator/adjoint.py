import cupy as cp
import numpy as np

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator


class Adjoint(Differentiator):
    def __call__(
        self, circuit: Circuit, final_state_vector: np.ndarray, params, expectation_op
    ):
        bp_circuit = self._get_bp_circuit(circuit)
        # prepare simulator
        self._simulator.initial_circuit(bp_circuit)
        self._simulator.vector = self._simulator._array_helper.array(
            final_state_vector, dtype=self._precision
        )
        # the blue path |psi_t> = U'|psi_t+1>
        current_state_vector = self._simulator.vector
        for gate in bp_circuit.gates:
            self._simulator.apply_gate(gate)
            later_state_vector = current_state_vector
            current_state_vector = self._simulator.vector

    def _get_bp_circuit(self, circuit):
        bp_circuit = Circuit(circuit.width())
        gates = circuit.gates[::-1]
        for gate in gates:
            inverse_gate = gate.inverse()
            inverse_gate | bp_circuit
        return bp_circuit


if __name__ == "__main__":
    circuit = Circuit(3)
    H | circuit(2)
    CRz(0.3) | circuit([2, 1])

    bp_circuit = Circuit(circuit.width())
    gates = circuit.gates[::-1]
    for gate in gates:
        inverse_gate = gate.inverse()
        inverse_gate | bp_circuit

    for gate in bp_circuit.gates:
        print(gate)

