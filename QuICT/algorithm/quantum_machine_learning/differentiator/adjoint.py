import cupy as cp
import numpy as np

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit

from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator


class Adjoint(Differentiator):
    def __call__(
        self, circuit: Circuit, final_state_vector: np.ndarray, expectation_op=Z
    ):
        # construct circuit for the blue path (1) and the orange path (2)
        bp_circuit1, bp_circuit2 = self._get_bp_circuit(circuit)
        gates1 = bp_circuit1.gates
        gates2 = bp_circuit2.gates
        n_qubits = bp_circuit1.width()
        
        current_state_grad = final_state_vector
        current_state_vector = final_state_vector
        for gate1, gate2 in zip(gates1, gates2):
            # d(L)/d(|psi_t>)
            current_state_grad = self._simulator.apply_gate(
                gate1, gate1.cargs + gate1.targs, current_state_grad, n_qubits
            )
            # |psi_t-1>
            current_state_vector = self._simulator.apply_gate(
                gate2, gate2.cargs + gate2.targs, current_state_vector, n_qubits
            )
            if self._is_param_gate(gate2):
                # d(L) / d(gate2)
                L_gate2_grad = current_state_grad @ current_state_vector.T
            else:
                continue

    def _is_param_gate(self, gate):
        if gate.params == 0:
            return False
        for parg in gate.pargs:
            if isinstance(parg, Variable):
                return True
        return False
    
    def _get_bp_circuits(self, circuit, expectation_op=Z):
        bp_circuit1 = Circuit(circuit.width())
        bp_circuit2 = Circuit(circuit.width())
        gates = circuit.gates[::-1]
        expectation_op | bp_circuit1

        for i in range(len(gates)):
            inverse_gate = gates[i].inverse()
            if i < len(gates) - 1:
                inverse_gate | bp_circuit1
            inverse_gate | bp_circuit2

        return bp_circuit1, bp_circuit2


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

