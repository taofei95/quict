import numpy as np
import logging
from typing import Union, List, Dict
import matplotlib.pyplot as plt

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition
from QuICT.qcda.optimization import CommutativeOptimization
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_algorithm.quantum_walk import Graph
from QuICT.algorithm.quantum_algorithm.quantum_walk import QuantumWalk


class QuantumWalkSearch(QuantumWalk):
    """ Grover algorithm based on quantum walk.

        https://arxiv.org/pdf/quant-ph/0210064.pdf
    """

    def __init__(self, simulator=ConstantStateVectorSimulator()):
        QuantumWalk.__init__(self, simulator)
        self._target = None
        self._coin_marked = None
        self._coin_unmarked = None
        self._coin_oracle = None
        self.sv = None

    def _coin_oracle_validation(self, coin_oracle):
        shape = coin_oracle.shape
        log2_shape = int(np.ceil(np.log2(shape[0])))

        return (
                shape[0] == shape[1] == 1 << self._total_qubits and
                shape[0] == (1 << log2_shape) and
                np.allclose(np.eye(shape[0]), coin_oracle.dot(coin_oracle.T.conj()))
        )

    def _circuit_construct(self):
        """ Construct random walk circuit. """
        # Build Circuit
        self._circuit = Circuit(self._total_qubits)
        for idx in range(self._total_qubits):
            H | self.circuit(idx)
        for t in range(self.step):
            self._coin_operator | self._circuit
            self._build_shift_operator() | self._circuit

    def _build_coin_operator(self, an=5 / 8, a0=1 / 8):
        if self._coin_unmarked is None:  # set C0 to G
            s_c = np.ones((2 ** self._action_qubits, 2 ** self._action_qubits)) / (2 ** self._action_qubits)
            self._coin_unmarked = np.eye(2 ** self._action_qubits) - 2 * s_c
        if self._coin_marked is None:
            x = np.zeros((1, 2 ** self._action_qubits))
            a = np.sqrt(an ** 2 + (a0 ** 2) * (2 ** self._action_qubits - 1))
            for i in range(2 ** self._action_qubits):
                if i == self._position_qubits - 1:
                    x[0, i] = an / a
                else:
                    x[0, i] = a0 / a
            self._coin_marked = np.eye(2 ** self._action_qubits) - 2 * (x.T @ x)
        search_array = np.zeros((self._graph.position, self._graph.position))
        search_array[self._target][self._target] = 1
        coin_oracle = np.kron(np.eye(self._graph.position), self._coin_unmarked) + np.kron(search_array, self._coin_marked - self._coin_unmarked)
        return Unitary(coin_oracle)
        # operators = []
        # for i in range(self._graph.position):
        #     if i == self._target:
        #         operators.append(coin_marked)
        #     else:
        #         operators.append(coin_unmarked)
        # return operators

    def _is_unit_hamming_distance(self, x, y):
        return str(bin(x ^ y))[2:].count("1") == 1

    def _get_hypercube_edges(self, position):
        edge = []
        for i in range(position):
            e = list(np.zeros(self._position_qubits, dtype=np.int64))
            for j in range(position):
                if self._is_unit_hamming_distance(i, j):
                    e[int(np.ceil(np.log2(i ^ j)))] = j
            edge.append(e)
        return edge

    def draw(self):
        p = self.sv.real * self.sv.real
        prob = np.zeros(self._graph.position)
        idx = 0
        for i in range(0, 1 << self._total_qubits, 2 ** self._action_qubits):
            for j in range(self._position_qubits):
                prob[idx] += p[i + j]
            idx += 1
        plt.bar(range(self._graph.position), prob)
        plt.show()

    def run(self,
            index_qubits: int,
            target: int = None,
            step: int = None,
            coin_marked: np.ndarray = None,
            coin_unmarked: np.ndarray = None,
            coin_oracle: np.ndarray = None,
            switched_time: int = -1,
            optimization: bool = False,
            record_measured: bool = False,
            ):

        self._position_qubits = index_qubits  # n
        self._action_qubits = int(np.ceil(np.log2(index_qubits)))  # c
        self._total_qubits = self._position_qubits + self._action_qubits
        position = 1 << index_qubits  # N
        self._step = step if step is not None and step > 0 else int(np.ceil(np.sqrt(position) * np.pi / 2)) + 1
        self._coin_marked = coin_marked
        self._coin_unmarked = coin_unmarked
        edges = self._get_hypercube_edges(position)
        self._graph = Graph(position, edges, None, switched_time)

        # Validation graph
        assert self._graph.validation(), "The edge's number should be equal."
        # Validation coin operator
        assert coin_oracle is not None or target is not None, "Should provide a coin oracle or a target index."
        if target is not None:
            assert 0 <= target < position, "Target should be within the range of values allowed by the index register. "
            self._target = target
        if coin_oracle is not None:
            assert self._coin_oracle_validation(coin_oracle), "The coin oracle should be a unitary matrix with side " \
                                                              "length 2 ** totel_qubits. "
            self._coin_operator = Unitary(coin_oracle)
        else:
            self._coin_operator = self._build_coin_operator()

        # Build random walk circuit
        self._circuit_construct()

        # Step 1, transform the unitary gate and optimization
        if optimization:
            opt_circuit = GateDecomposition.execute(self._circuit)
            opt_circuit = CommutativeOptimization.execute(opt_circuit)
        else:
            opt_circuit = self._circuit

        # Return final state vector if not need
        if not record_measured:
            self.sv = self._simulator.run(opt_circuit)
            return self.sv


if __name__ == "__main__":
    from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator

    simulator = ConstantStateVectorSimulator()
    grover = QuantumWalkSearch(simulator)
    result = grover.run(index_qubits=2, target=2)
    grover.draw()
