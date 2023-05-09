import numpy as np
from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator


class ParameterShift:
    __DEVICE = ["CPU", "GPU"]
    __PRECISION = ["single", "double"]

    def __init__(
        self,
        device: str = "GPU",
        precision: str = "double",
        gpu_device_id: int = 0,
        sync: bool = True,
    ):
        if device not in self.__DEVICE:
            raise ValueError("ParameterShift.device", "[CPU, GPU]", device)

        if precision not in self.__PRECISION:
            raise ValueError("ParameterShift.precision", "[single, double]", precision)

        self._device = device
        self._precision = precision
        self._device_id = gpu_device_id
        self._sync = sync
        self._training_gates = 0
        self._remain_training_gates = 0
        self._simulator = StateVectorSimulator(
            self._device, self._precision, self._device_id, self._sync
        )

    def run(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_op: Hamiltonian,
    ):
        self._initial_circuit(circuit)
        expectation = self.get_expectation(state_vector, expectation_op)
        for gate in self._pipeline:
            if self._remain_training_gates == 0:
                return variables, expectation
            if isinstance(gate, BasicGate):
                self._calculate_grad(variables, gate, expectation_op)
            else:
                raise TypeError("ParameterShift.run.circuit", "BasicGate".type(gate))

        return variables, expectation

    def get_expectation(
        self, state_vector: np.ndarray, expectation_op: Hamiltonian,
    ):
        circuit_list = expectation_op.construct_hamiton_circuit(self._qubits)
        coefficients = expectation_op.coefficients
        grad_vector = np.zeros(1 << self._qubits, dtype=np.complex128)
        grad_vector = self._simulator._gate_calculator.normalized_state_vector(
            grad_vector, self._qubits
        )
        for coeff, circuit in zip(coefficients, circuit_list):
            grad_vec = self._simulator.run(circuit, state_vector)
            grad_vector += coeff * grad_vec

        expectation = (
            (state_vector.conj() @ grad_vector).real
            if self._device == "CPU"
            else (state_vector.conj() @ grad_vector).real.get()
        )
        return expectation

    def _calculate_grad(self, variables: Variable, gate, expectation_op):
        if gate.variables == 0:
            return
        self._remain_training_gates -= 1
        for i in range(gate.params):
            if isinstance(gate.pargs[i], Variable):
                origin_param = gate.pargs[i].copy()
                gate.pargs[i] += np.pi / 2
                state_vector_fw = self._simulator.run(self._circuit)
                expectation_fw = self.get_expectation(state_vector_fw, expectation_op)
                gate.pargs[i] -= np.pi
                state_vector_bw = self._simulator.run(self._circuit)
                expectation_bw = self.get_expectation(state_vector_bw, expectation_op)
                gate.pargs[i] = origin_param
                grad = (expectation_fw - expectation_bw) / 4
                gate.pargs[i].grads = (
                    grad
                    if abs(gate.pargs[i].grads) < 1e-12
                    else grad * gate.pargs[i].grads
                )
                index = gate.pargs[i].index
                variables.grads[index] += gate.pargs[i].grads

    def _initial_circuit(self, circuit: Circuit):
        circuit.gate_decomposition(decomposition=False)
        self._training_gates = circuit.count_training_gate()
        self._remain_training_gates = self._training_gates
        self._qubits = int(circuit.width())
        self._circuit = circuit
        self._pipeline = [gate & targs for gate, targs, _ in circuit.fast_gates]
