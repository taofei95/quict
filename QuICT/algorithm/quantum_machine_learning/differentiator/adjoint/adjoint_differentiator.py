import numpy as np

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.utils import GateSimulator
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian


class AdjointDifferentiator:
    __DEVICE = ["CPU", "GPU"]
    __PRECISION = ["single", "double"]

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, circuit):
        self._circuit = circuit

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        self._vector = self._gate_calculator.validate_state_vector(vec, self._qubits)

    @property
    def device(self):
        return self._device_id

    @property
    def grad_vector(self):
        return self._grad_vector

    @grad_vector.setter
    def grad_vector(self, vec):
        self._grad_vector = self._gate_calculator.normalized_state_vector(
            vec, self._qubits
        )

    def __init__(
        self,
        device: str = "GPU",
        precision: str = "double",
        gpu_device_id: int = 0,
        sync: bool = True,
    ):
        if device not in self.__DEVICE:
            raise ValueError("AdjointDifferentiator.device", "[CPU, GPU]", device)

        if precision not in self.__PRECISION:
            raise ValueError(
                "AdjointDifferentiator.precision", "[single, double]", precision
            )

        self._device = device
        self._precision = precision
        self._device_id = gpu_device_id
        self._sync = sync
        self._training_gates = 0
        self._remain_training_gates = 0
        self._gate_calculator = GateSimulator(
            self._device, self._precision, self._device_id, self._sync
        )

    def run(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_op: Hamiltonian,
    ):
        self.initial_circuit(circuit)
        assert state_vector is not None
        self._vector = self._gate_calculator.normalized_state_vector(
            state_vector, self._qubits
        )
        # Calculate d(L)/d(|psi_t>)
        self._grad_vector = self._initial_grad_vector(
            state_vector, self._qubits, expectation_op
        )

        for idx in range(len(self._bp_pipeline)):
            if self._remain_training_gates == 0:
                return variables
            origin_gate = self._pipeline[idx]
            gate, qidxes, _ = self._bp_pipeline[idx]
            if isinstance(gate, BasicGate):
                # Calculate |psi_t-1>
                self._apply_gate(gate, qidxes, self._vector)

                # Calculate d(L)/d(theta) and write to origin_gate.gate.pargs.grads
                self._calculate_grad(variables, origin_gate, qidxes)

                # Calculate d(L)/d(|psi_t-1>)
                self._apply_gate(gate, qidxes, self._grad_vector)
            else:
                raise TypeError(
                    "AdjointDifferentiator.run.circuit", "BasicGate".type(gate)
                )
        return variables

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
        grad_vector = self._gate_calculator.normalized_state_vector(grad_vector, qubits)
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

    def _calculate_grad(self, variables: Variable, origin_gate, qidxes: list):
        if origin_gate.variables > 0:
            self._remain_training_gates -= 1
        for i in range(origin_gate.variables):
            vector = self._vector.copy()
            # d(|psi_t>) / d(theta_t^j)
            self._apply_gate(origin_gate, qidxes, vector, fp=False, parg_id=i)

            # d(L)/d(|psi_t>) * d(|psi_t>) / d(theta_t^j)
            grad = np.float64((self._grad_vector @ vector.conj()).real)

            # write gradient
            origin_gate.pargs[i].grads = (
                grad
                if abs(origin_gate.pargs[i].grads) < 1e-12
                else grad * origin_gate.pargs[i].grads
            )
            index = origin_gate.pargs[i].index
            variables.grads[index] += origin_gate.pargs[i].grads


if __name__ == "__main__":
    from QuICT.simulation.state_vector import StateVectorSimulator

    param = Variable(np.array([-3.2]))

    circuit = Circuit(2)
    H | circuit
    Rxx(param[0]) | circuit([0, 1])

    simulator = StateVectorSimulator()
    sv = simulator.run(circuit)

    differ = AdjointDifferentiator(device="GPU")
    h = Hamiltonian([[1, "X1"]])
    differ.run(circuit, param, sv, h)
    print(param.grads)
