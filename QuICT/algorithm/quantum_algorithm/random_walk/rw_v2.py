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

    def __init__(self, T: int, graph: Graph, coin_operator: np.ndarray):
        self._step =  T
        self._graph = graph
        self._coin_operator = coin_operator

        # Validation graph and coin operator
        self._graph_validation()
        self._cop_validation()
        self._total_qubits = self._position_qubits + self._action_qubits

        # Build random walk circuit
        self._circuit_construct()
        
    def _graph_validation(self):
        self._position_space = self._graph.vector
        self._position_qubits = int(np.ceil(np.log2(self._position_space)))
        self._edge_dict = self._graph.edges
        self._action_space = len(self._edge_dict[0])
        assert self._graph.validation(), "The edge's number should be equal."

    def _cop_validation(self):
        shape = self._coin_operator.shape
        print(shape)
        assert shape[0] == shape[1] and shape[0] >= self._action_space
        assert np.allclose(np.eye(shape[0]), self._coin_operator.dot(self._coin_operator.T.conj()))

        self._action_qubits = int(np.ceil(np.log2(shape[0])))

    def _circuit_construct(self):
        self._circuit = Circuit(self._total_qubits)
        action_gate = Unitary(self._coin_operator)
        action_qubits = [self._position_qubits + i for i in range(self._action_qubits)]

        for _ in range(self.step):
            action_gate | self._circuit(action_qubits)
            self._build_shift_operator() | self._circuit

    def _build_shift_operator(self):
        unitary_matrix = np.zeros((1 << self._total_qubits, 1 << self._total_qubits), dtype = np.complex128)
        for i in range(self._position_space):
            curr_idx = (1 << self._action_qubits) * i
            for action_state in range(self._action_space):
                related_action = self._edge_dict[i][action_state]
                unitary_matrix[related_action * (1 << self._action_qubits) + action_state, curr_idx + action_state] = 1

        return Unitary(unitary_matrix)

    def run(self, device: str = "GPU", record_measured: bool = False):
        # TODO: bug fixed linalg.cpu.dot.
        pass
