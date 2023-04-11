import numpy as np

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian


class Adjoint(Differentiator):
    @property
    def grad_vector(self):
        return self._grad_vector

    @grad_vector.setter
    def grad_vector(self, vec):
        self._grad_vector = self._gate_calculator.validate_state_vector(
            vec, self._qubits
        )

    def run(
        self, circuit: Circuit, state_vector: np.ndarray, expectation_op: Hamiltonian,
    ) -> np.ndarray:
        self.initial_circuit(circuit)
        assert state_vector is not None
        self._vector = self._gate_calculator.validate_state_vector(
            state_vector, self._qubits
        )
        # Calculate d(L)/d(|psi_t>)
        self._grad_vector = self._initial_grad_vector(
            state_vector, self._qubits, expectation_op
        )

        for idx in range(len(self._bp_pipeline)):
            if self._remain_training_gates == 0:
                return
            origin_gate = self._pipeline[idx]
            gate, qidxes, _ = self._bp_pipeline[idx]
            if isinstance(gate, BasicGate):
                # Calculate |psi_t-1>
                self._apply_gate(gate, qidxes, self._vector)

                # Calculate d(L)/d(theta) and write to origin_gate.gate.pargs.grads
                self._calculate_grad(origin_gate, gate, qidxes)

                # Calculate d(L)/d(|psi_t-1>)
                self._apply_gate(gate, qidxes, self._grad_vector)
            else:
                raise TypeError("Adjoint.run.circuit", "BasicGate".type(gate))

    def initial_circuit(self, circuit: Circuit):
        circuit.gate_decomposition(decomposition=False)
        self._training_gates = circuit.count_training_gates()
        self._remain_training_gates = self._training_gates
        self._qubits = int(circuit.width())
        self._circuit = circuit
        self._bp_circuit = Circuit(circuit.width())
        gates = circuit.gates[::-1]
        for i in range(len(gates)):
            inverse_gate = gates[i].inverse()
            inverse_gate.targs = gates[i].targs
            inverse_gate.cargs = gates[i].cargs
            inverse_gate | self._bp_circuit
        self._pipeline = gates
        self._bp_pipeline = self._bp_circuit.fast_gates
        assert len(self._pipeline) == len(self._bp_pipeline)

    # optimize? simulator x
    def _initial_grad_vector(
        self, state_vector, qubits: int, expectation_op: Hamiltonian
    ):
        state_vector_copy = state_vector.copy()
        simulator = StateVectorSimulator(
            self._device, self._precision, self._device_id, self._sync
        )

        circuit_list = expectation_op.construct_hamiton_circuit(qubits)
        coefficients = expectation_op.coefficients
        grad_vector = np.zeros(1 << qubits, dtype=np.complex128)
        grad_vector = self._gate_calculator.validate_state_vector(grad_vector, qubits)
        for coeff, circuit in zip(coefficients, circuit_list):
            grad_vec = simulator.run(circuit, state_vector_copy)
            grad_vector += coeff * grad_vec

        return grad_vector

    def _apply_gate(
        self, gate: BasicGate, qidxes: list, vector, fp: bool = True, parg_id: int = 0
    ):
        gate_type = gate.type
        if gate_type in [GateType.measure, GateType.reset]:
            raise NotImplementedError
        else:
            self._gate_calculator.apply_gate(
                gate, qidxes, vector, self._qubits, fp, parg_id
            )

    def _calculate_grad(self, origin_gate, gate, qidxes: list):
        if gate.variables > 0:
            self._remain_training_gates -= 1
        for i in range(gate.variables):
            vector = self._vector.copy()
            # d(|psi_t>) / d(theta_t^j)
            self._apply_gate(origin_gate, qidxes, vector, fp=False, parg_id=i)

            # d(L)/d(|psi_t>) * d(|psi_t>) / d(theta_t^j)
            grad = np.float64((self._grad_vector @ vector.conj()).real)
            origin_gate.pargs[i].grads = grad


if __name__ == "__main__":
    from QuICT.core.gate.utils import Variable
    from QuICT.simulation.utils import GateSimulator
    from QuICT.simulation.state_vector import StateVectorSimulator

    param = Variable(np.array([-3.2]))

    circuit = Circuit(2)
    H | circuit
    Rxx(param[0]) | circuit([0, 1])

    simulator = StateVectorSimulator()
    sv = simulator.run(circuit)

    differ = Adjoint(device="GPU")
    h = Hamiltonian([[1, "X1"]])
    differ.run(circuit, sv, h)
