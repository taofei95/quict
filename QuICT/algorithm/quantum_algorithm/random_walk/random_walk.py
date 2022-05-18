import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
import QuICT.ops.linalg.cpu_calculator as linalg
from .graph import Graph


class RandomWalk:
    @property
    def step(self):
        return self._step

    @property
    def graph(self):
        return str(self._graph)

    @property
    def circuit(self) -> Circuit:
        return self._circuit

    def __init__(self, T: int, graph: Graph, coin_operator: np.ndarray = None):
        self._step = T
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
        if self._coin_operator is not None:
            assert self._graph.operator_validation(self._coin_operator), "The operator should be an unitary matrix."
            self._action_qubits = int(np.ceil(np.log2(self._coin_operator.shape[0])))
        else:
            self._coin_operator = self._graph.operators
            self._action_qubits = self._graph.action_qubits
            self._operator_by_position = True
            self._operator_by_time = self._graph.switched_time > 0

    def _circuit_construct(self):
        self._circuit = Circuit(self._total_qubits)
        for t in range(self.step):
            self._build_action_operator(t) | self._circuit
            self._build_shift_operator() | self._circuit

    def _build_action_operator(self, step: int):
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

        return action_gate

    def _mct_generator(self, op: np.ndarray):
        mct_unitary = np.identity(1 << self._total_qubits, dtype=np.complex128)
        op_shape = op.shape
        mct_unitary[-op_shape[0]:, -op_shape[1]:] = op

        return Unitary(mct_unitary) & list(range(self._total_qubits))

    def _build_shift_operator(self):
        unitary_matrix = np.zeros((1 << self._total_qubits, 1 << self._total_qubits), dtype=np.complex128)
        for i in range(self._graph.position):
            curr_idx = (1 << self._action_qubits) * i
            for action_state in range(self._graph.action_space):
                related_action = self._graph.edges[i][action_state]
                unitary_matrix[related_action * (1 << self._action_qubits) + action_state, curr_idx + action_state] = 1

        return Unitary(unitary_matrix)

    def run(self, device: str = "GPU", record_measured: bool = False):
        # TODO: bug fixed linalg.cpu.dot.
        pass
