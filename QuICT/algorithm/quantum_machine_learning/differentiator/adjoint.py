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
        grad_circuit, uncompute_circuit = self._get_bp_circuits(circuit, expectation_op)
        gates = circuit._gates[::-1]
        gates1 = grad_circuit._gates
        gates2 = uncompute_circuit._gates
        n_qubits = grad_circuit.width()

        current_state_grad = final_state_vector
        current_state_vector = final_state_vector

        for gate, gate1, gate2 in zip(gates, gates1, gates2):
            # d(L)/d(|psi_t>)
            # NEED TO REWRITE SIMULATOR
            current_state_grad = self._simulator.apply_gate(
                gate1, gate1.cargs + gate1.targs, current_state_grad, n_qubits
            )
            # |psi_t-1>
            current_state_vector = self._simulator.apply_gate(
                gate2, gate2.cargs + gate2.targs, current_state_vector, n_qubits
            )
            if gate.variables > 0:
                for i in range(gate.variables):
                    parg_grad = self._simulator.apply_gate(
                        gate,
                        gate.cargs + gate.targs,
                        current_state_vector,
                        n_qubits,
                        fp=False,
                        parg_id=i,
                    )
                    grad = current_state_grad @ parg_grad.T
                    gate.pargs[i].grad = grad
            else:
                continue

    def _get_bp_circuits(self, circuit, expectation_op=Z):
        grad_circuit = Circuit(circuit.width())
        uncompute_circuit = Circuit(circuit.width())
        gates = circuit.gates[::-1]
        Z | grad_circuit(2)

        for i in range(len(gates)):
            inverse_gate = gates[i].inverse()
            inverse_gate.targs = gates[i].targs
            inverse_gate.cargs = gates[i].cargs

            if i < len(gates) - 1:
                inverse_gate | grad_circuit
            inverse_gate | uncompute_circuit

        return grad_circuit, uncompute_circuit


if __name__ == "__main__":
    from QuICT.core.gate.utils import Variable
    from QuICT.simulation.utils import GateSimulator
    from QuICT.simulation.state_vector import StateVectorSimulator

    param = Variable(np.array([0.3]))

    circuit = Circuit(3)
    H | circuit(2)
    Z | circuit(0)
    CRz(param[0]) | circuit([2, 1])
    # for gate, _, _ in circuit.fast_gates:
    #     print(gate.pargs)

    simulator = StateVectorSimulator()
    sv = simulator.run(circuit)

    differ = Adjoint(device="CPU")
    differ(circuit, sv)
    gates = circuit.gates
    for gate in gates:
        for parg in gate.pargs:
            if isinstance(parg, Variable):
                print(parg.grads)

