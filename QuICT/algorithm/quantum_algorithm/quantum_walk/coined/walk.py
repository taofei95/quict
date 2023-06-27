import numpy as np
from typing import Dict, List, Union

from ..graph import Graph
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator


class Walk:

    __COIN = ["H", "Grover"]

    def __init__(self, simulator=StateVectorSimulator()):
        """ Initialize the simulator circuit of quantum random walk.

        Args:
            simulator (Union[StateVectorSimulator, StateVectorSimulator], optional):
                The simulator for simulating quantum circuit. Defaults to StateVectorSimulator().
        """
        self._simulator = simulator
        self._node_qubits = None
        self._coin_qubits = None
        self._n_qubits = None

    def _coin_operator(self, coin) -> CompositeGate:
        """ Generator coin operator. """
        coin_operator = CompositeGate(self._coin_qubits)
        
        action_qubits = [self._position_qubits + i for i in range(self._action_qubits)]
        if not (self._operator_by_position or self._operator_by_time):
            return Unitary(self._coin_operator) & action_qubits

    def _shift_operator(self):
        """ Generator shift operator. """
        unitary_matrix = np.zeros(
            (1 << self._total_qubits, 1 << self._total_qubits), dtype=np.complex128
        )
        record_idxes = list(range(1 << self._total_qubits))
        for i in range(self._graph.position):
            curr_idx = (1 << self._action_qubits) * i
            for action_state in range(self._graph.action_space):
                related_idx = self._graph.edges[i][action_state] * (
                    1 << self._action_qubits
                )
                unitary_matrix[related_idx + action_state, curr_idx + action_state] = 1
                if related_idx + action_state in record_idxes:
                    record_idxes.remove(related_idx + action_state)

        if len(record_idxes) > 0:
            unitary_matrix[record_idxes, record_idxes] = 1

        return Unitary(unitary_matrix)

    def run(
        self,
        step: int,
        nodes: int,
        edges: Union[List, Dict] = None,
        coin: str = "H",
        shots: int = 1000,
    ) -> Union[np.ndarray, List]:
        assert coin in Walk.__COIN
        self._node_qubits = int(np.ceil(np.log2(nodes)))
        self._coin_qubits = 
        
        # Build random walk circuit
        circuit = Circuit(self._n_qubits)
        for t in range(step):
            self._coin_operator(coin) | self._circuit
            self._shift_operator() | self._circuit

        # Simulate the quantum walk's circuit
        _ = self._simulator.run(self._circuit)

        return self._simulator.sample(shots)
