import numpy as np
from typing import Union, List

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.unitary_decomposition import UnitaryDecomposition
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition
from QuICT.qcda.optimization import CommutativeOptimization
from QuICT.simulation.state_vector import CircuitSimulator
from .graph import Graph


class QuantumWalk:
    """ The Quantum Random Walk Algorithm """
    @property
    def step(self):
        return self._step

    @property
    def graph(self):
        return str(self._graph)

    @property
    def circuit(self) -> Circuit:
        """ The quantum circuit of the random walk algorithm, including UnitaryGate """
        return self._circuit

    def __init__(self, step: int, graph: Graph, coin_operator: np.ndarray = None):
        """ Initial the quantum random walk with given steps, graph and coin operator.

        Args:
            T (int): The steps of random walk, a step including a coin operator and a shift operator.
            graph (Graph): The description of the position state, vectors represent the position space and
                the edges represent the action space.
            coin_operator (np.ndarray, optional): The coin operators, the unitary matrix. Defaults to None.
        """
        self._step = step
        self._graph = graph
        self._coin_operator = coin_operator
        self._operator_by_position = False
        self._operator_by_time = False

        # Validation graph and coin operator
        assert self._graph.validation(), "The edge's number should be equal."
        self._operator_validation()
        self._total_qubits = self._graph.position_qubits + self._action_qubits

        # Build random walk circuit
        self._circuit_construct()

    def _operator_validation(self):
        """ Validate the operator """
        if self._coin_operator is not None:
            assert self._graph.operator_validation(self._coin_operator), "The operator should be an unitary matrix."
            self._action_qubits = int(np.ceil(np.log2(self._coin_operator.shape[0])))
        else:
            self._coin_operator = self._graph.operators
            self._action_qubits = self._graph.action_qubits
            self._operator_by_position = True
            self._operator_by_time = self._graph.switched_time > 0

    def _circuit_construct(self):
        """ Construct random walk circuit """
        # Build shift operator
        self._build_shift_operator()

        # Build Circuit
        self._circuit = Circuit(self._total_qubits)
        for t in range(self.step):
            self._build_action_operator(t) | self._circuit
            self._shift_operator | self._circuit

    def _build_action_operator(self, step: int) -> CompositeGate:
        """ Generator action operator """
        action_qubits = [self._graph.position_qubits + i for i in range(self._action_qubits)]
        if not (self._operator_by_position or self._operator_by_time):
            return Unitary(self._coin_operator) & action_qubits

        action_gate = CompositeGate()
        curr_op_idx = (step // self._graph.switched_time) % len(self._graph.operators[0]) \
            if self._operator_by_time else 0
        for i in range(self._graph.position):
            op = self._graph.operators[i][curr_op_idx]
            x_idx = [pidx for pidx in range(self._graph.position_qubits) if not (i & 1 << pidx)]
            for xi in x_idx:
                X | action_gate(xi)

            self._mct_generator(op) | action_gate
            for xi in x_idx:
                X | action_gate(xi)

        return GateDecomposition.execute(action_gate)

    def _mct_generator(self, op: np.ndarray) -> UnitaryGate:
        """ Build multi-control-'op' gate """
        mct_unitary = np.identity(1 << self._total_qubits, dtype=np.complex128)
        op_shape = op.shape
        mct_unitary[-op_shape[0]:, -op_shape[1]:] = op

        return Unitary(mct_unitary) & list(range(self._total_qubits))

    def _build_shift_operator(self):
        """ Generator shift operator """
        unitary_matrix = np.zeros((1 << self._total_qubits, 1 << self._total_qubits), dtype=np.complex128)
        record_idxes = list(range(1 << self._total_qubits))
        for i in range(self._graph.position):
            curr_idx = (1 << self._action_qubits) * i
            for action_state in range(self._graph.action_space):
                related_idx = self._graph.edges[i][action_state] * (1 << self._action_qubits)
                unitary_matrix[related_idx + action_state, curr_idx + action_state] = 1
                if related_idx + action_state in record_idxes:
                    record_idxes.remove(related_idx + action_state)

        if len(record_idxes) > 0:
            unitary_matrix[record_idxes, record_idxes] = 1

        self._shift_operator = UnitaryDecomposition().execute(unitary_matrix)[0]

    def run(
        self,
        simulator=CircuitSimulator(),
        optimization: bool = False,
        record_measured: bool = False,
        shots: int = 1
    ) -> Union[np.ndarray, List]:
        """ Simulate the quantum random walk circuit.

        Args:
            simulator (Union[ConstantStateVectorSimulator, CircuitSimulator], optional):
                The simulator for simulating quantum circuit. Defaults to CircuitSimulator().
            optimization (bool, optional): whether using QCDA to optimize quantum walk circuit, may
                spend lots of times when circuit is large.
            record_measured (bool, optional): whether return the final measured state with shots time,
                or return the state vector after simulating. Defaults to False.
            shots (int, optional): The repeatted times. Defaults to 1.

        Returns:
            Union[np.ndarray, List]: The state vector or measured states
        """
        # Step 1, transform the unitary gate and optimization
        if optimization:
            opt_circuit = GateDecomposition.execute(self._circuit)
            opt_circuit = CommutativeOptimization.execute(opt_circuit)
        else:
            opt_circuit = self._circuit

        # Step 2, Simulate the quantum walk's circuit
        state_vector = simulator.run(self.circuit)

        # Return final state vector if not need
        if not record_measured:
            return state_vector

        return simulator.sample(shots)
