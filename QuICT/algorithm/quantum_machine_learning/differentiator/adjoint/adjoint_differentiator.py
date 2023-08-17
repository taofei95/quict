import numpy as np

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.core.circuit import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.simulation.utils import GateSimulator


class AdjointDifferentiator:
    """The differentiator using adjoint method.

    References:
    https://arxiv.org/abs/1912.10877
    """

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
        """Initialize a adjoint differentiator.

        Args:
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "GPU".
            precision (str, optional): The precision for the state vector, one of [single, double].
                Defaults to "double".
            gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
            sync (bool, optional): Sync mode or Async mode. Defaults to True.
        """

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
        self._simulator = StateVectorSimulator(
            self._device, self._precision, self._device_id, self._sync
        )

    def _run_one_op(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_op: Hamiltonian,
    ):
        """Calculate the gradients and expectation of a Parameterized Quantum Circuit (PQC).

        Args:
            circuit (Circuit): PQC that needs to calculate gradients.
            variables (Variable): The parameters of the circuit.
            state_vector (np.ndarray): The state vector output from forward propagation.
            expectation_op (Hamiltonian): The hamiltonian that need to get expectation.

        Returns:
            np.ndarry: The gradients of parameters (params_shape).
            np.float: The expectation.
        """

        self._initial_circuit(circuit)
        self._vector = state_vector.copy()

        # Calculate d(L)/d(|psi_t>)
        self._grad_vector = 2.0 * self._initial_grad_vector(
            state_vector.copy(), self._qubits, expectation_op
        )
        # expectation
        expectation = (
            (state_vector.conj() @ (self._grad_vector / 2.0)).real
            if self._device == "CPU"
            else (state_vector.conj() @ (self._grad_vector / 2.0)).real.get()
        )

        for idx in range(len(self._bp_pipeline)):
            if self._remain_training_gates == 0:
                return variables.grads, expectation
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
        return variables.grads, expectation

    def run(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_ops: list,
    ):
        """Calculate the gradients and expectation of a Parameterized Quantum Circuit (PQC).

        Args:
            circuit (Circuit): PQC that needs to calculate gradients.
            variables (Variable): The parameters of the circuit.
            state_vector (np.ndarray): The state vector output from forward propagation.
            expectation_ops (list): The hamiltonians that need to get expectations.

        Returns:
            np.ndarray: The gradients of parameters (ops_num, params_shape).
            np.ndarray: The expectations (ops_num, ).
        """
        params_grad_list = []
        expectation_list = []
        for op in expectation_ops:
            params_grad, expectation = self._run_one_op(
                circuit, variables.copy(), state_vector, op
            )
            params_grad_list.append(params_grad)
            expectation_list.append(expectation)

        return np.array(params_grad_list), np.array(expectation_list)

    def run_batch(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector_list: list,
        expectation_ops: list,
    ):
        """Calculate the gradients and expectations of a batch of PQCs.

        Args:
            circuit (Circuit): PQC that needs to calculate gradients.
            variables (Variable): The parameters of the circuit.
            state_vector_list (list): The state vectors output from multiple FP process.
            expectation_ops (list): The hamiltonians that need to get expectations.

        Returns:
            np.ndarray: The gradients of parameters (batch_size, ops_num, params_shape).
            np.ndarray: The expectations (batch_size, ops_num).
        """
        params_grad_list = []
        expectation_list = []
        for state_vector in state_vector_list:
            params_grads, expectations = self.run(
                circuit, variables.copy(), state_vector, expectation_ops
            )
            params_grad_list.append(params_grads)
            expectation_list.append(expectations)

        return np.array(params_grad_list), np.array(expectation_list)

    def _get_one_expectation(
        self, circuit: Circuit, state_vector: np.ndarray, expectation_op: Hamiltonian,
    ):
        """Calculate the expectation of a PQC.

        Args:
            circuit (Circuit): The PQC.
            state_vector (np.ndarray): The state vector output from forward propagation.
            expectation_op (Hamiltonian): The hamiltonian that need to get expectation.

        Returns:
            np.float: The expectation.
        """
        self._initial_circuit(circuit)

        # Calculate d(L)/d(|psi_t>)
        self._grad_vector = self._initial_grad_vector(
            state_vector.copy(), self._qubits, expectation_op
        )
        # expectation
        expectation = (
            (state_vector.conj() @ self._grad_vector).real
            if self._device == "CPU"
            else (state_vector.conj() @ self._grad_vector).real.get()
        )
        return expectation

    def get_expectations(
        self, circuit: Circuit, state_vector: np.ndarray, expectation_ops: list,
    ):
        """Calculate the expectation of a PQC.

        Args:
            circuit (Circuit): The PQC.
            state_vector (np.ndarray): The state vector output from forward propagation.
            expectation_ops (list): The hamiltonians that need to get expectations.

        Returns:
            np.ndarray: The expectations.
        """
        expectation_list = []
        for op in expectation_ops:
            expectation = self._get_one_expectation(circuit, state_vector, op)
            expectation_list.append(expectation)
        return np.array(expectation_list)

    def get_expectations_batch(
        self, circuit: Circuit, state_vector_list: list, expectation_ops: list,
    ):
        """Calculate the expectations of a batch of PQCs.

        Args:
            circuit (Circuit): The PQC.
            state_vector_list (list): The state vectors output from multiple FP process.
            expectation_ops (list): The hamiltonians that need to get expectations.

        Returns:
            np.ndarray: The expectations.
        """
        expectation_list = []
        for state_vector in state_vector_list:
            expectations = self.get_expectations(circuit, state_vector, expectation_ops)
            expectation_list.append(expectations)
        return np.array(expectation_list)

    def _initial_circuit(self, circuit: Circuit):
        circuit.gate_decomposition(decomposition=False)
        self._training_gates = circuit.count_training_gate()
        self._remain_training_gates = self._training_gates
        self._qubits = int(circuit.width())
        self._circuit = circuit
        self._bp_circuit = circuit.inverse()
        gates = [gate & targs for gate, targs, _ in circuit.fast_gates][::-1]
        self._pipeline = gates
        self._bp_pipeline = self._bp_circuit.fast_gates

    def _initial_grad_vector(
        self, state_vector, qubits: int, expectation_op: Hamiltonian
    ):
        circuit_list = expectation_op.construct_hamiton_circuit(qubits)
        coefficients = expectation_op.coefficients
        grad_vector = self._gate_calculator.get_empty_state_vector(qubits)
        for coeff, circuit in zip(coefficients, circuit_list):
            grad_vec = self._simulator.run(circuit, state_vector)
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
        if origin_gate.variables == 0:
            return
        self._remain_training_gates -= 1
        for i in range(origin_gate.params):
            if isinstance(origin_gate.pargs[i], Variable):
                vector = self._vector.copy()
                # d(|psi_t>) / d(theta_t^j)
                self._apply_gate(origin_gate, qidxes, vector, fp=False, parg_id=i)

                # d(L)/d(theta_t^j) = d(L)/d(|psi_t>) * d(|psi_t>)/d(theta_t^j)
                grad = np.float64((self._grad_vector @ vector.conj()).real)

                # write gradient
                gate_grads = (
                    grad
                    if abs(origin_gate.pargs[i].grads) < 1e-12
                    else grad * origin_gate.pargs[i].grads
                )
                index = origin_gate.pargs[i].index
                variables.grads[index] += gate_grads
