import logging
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np

from QuICT.algorithm.quantum_algorithm.quantum_walk import Graph, QuantumWalk
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.optimization import CommutativeOptimization
from QuICT.qcda.synthesis.gate_decomposition import GateDecomposition
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


class QuantumWalkSearch(QuantumWalk):
    """ Search algorithm on a hypercube based on quantum walk and Grover.

        https://arxiv.org/pdf/quant-ph/0210064.pdf
        http://dx.doi.org/10.4236/jqis.2015.51002
    """

    def __init__(self, simulator=ConstantStateVectorSimulator()):
        """ Initialize the simulator circuit of quantum random walk.

        Args:
            simulator (Union[ConstantStateVectorSimulator, CircuitSimulator], optional):
                The simulator for simulating quantum circuit. Defaults to ConstantStateVectorSimulator().
        """
        QuantumWalk.__init__(self, simulator)
        self._target = None
        self._coin_marked = None
        self._coin_unmarked = None
        self._coin_oracle = None
        self.sv = None

    def _coin_oracle_validation(self, coin_oracle):
        """ Validate the coin oracle. """
        shape = coin_oracle.shape
        log2_shape = int(np.ceil(np.log2(shape[0])))

        return (
                shape[0] == shape[1] == 1 << self._total_qubits and
                shape[0] == (1 << log2_shape) and
                np.allclose(np.eye(shape[0]), coin_oracle.dot(coin_oracle.T.conj()))
        )

    def _circuit_construct(self):
        """ Construct random walk search circuit. """
        # Build Circuit
        self._circuit = Circuit(self._total_qubits)
        for idx in range(self._total_qubits):
            H | self.circuit(idx)
        for t in range(self.step):
            self._coin_operator | self._circuit
            self._build_shift_operator() | self._circuit

    def _build_coin_operator(self, an=5 / 8, a0=1 / 8):
        """ Generate a coin oracle by using the unmarked and marked coin operators. """
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

    def _is_unit_hamming_distance(self, x, y):
        """ Calculate the hamming distance of two nodes of the n-cube. """
        return str(bin(x ^ y))[2:].count("1") == 1

    def _get_hypercube_edges(self, position):
        """ Get the edges of the n-cube graph. """
        edge = []
        for i in range(position):
            e = list(np.zeros(self._position_qubits, dtype=np.int64))
            for j in range(position):
                if self._is_unit_hamming_distance(i, j):
                    e[int(np.ceil(np.log2(i ^ j)))] = j
            edge.append(e)
        return edge

    def draw(self):
        """ Plot the probability distribution of the states. """
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
            record_measured: bool = True,
            ):
        """ Execute the quantum walk search with given number of index qubits.

        Args:
            index_qubits (int): The size of the node register.
            target (int, optional): The index of the target element.
            step (int, optional): The steps of random walk, a step including a coin operator and a shift operator.
            coin_marked (np.ndarray, optional): The coin operator for the target node. Should be a unitary matrix.
            coin_unmarked (np.ndarray, optional): The coin operator for other nodes except the target. Should be a unitary matrix.
                Defaults to Grover coin.
            coin_oracle (np.ndarray, optional): A coin operator which takes on the function of an oracle. Should be a unitary matrix.
            switched_time (int, optional): The number of steps of each coin operator in the vector.
                Defaults to -1, means not switch coin operator.
            optimization (bool, optional): whether using QCDA to optimize quantum walk circuit, may
                spend lots of times when circuit is large.
            record_measured (bool, optional): whether return the final measured state. Defaults to True.

        Returns:
            Union[np.ndarray, List]: The state vector or measured states.
        """

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
            opt_circuit = GateDecomposition().execute(self._circuit)
            opt_circuit = CommutativeOptimization().execute(opt_circuit)
        else:
            opt_circuit = self._circuit

        # Return final state vector if not need
        if not record_measured:
            self.sv = self._simulator.run(opt_circuit)
            return self.sv


if __name__ == "__main__":
    simulator = ConstantStateVectorSimulator()
    grover = QuantumWalkSearch(simulator)
    result = grover.run(index_qubits=3, target=6)
    grover.draw()
