from __future__ import annotations

from collections import deque
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from QuICT.core import Circuit
from QuICT.core.gate.gate import *

from .utility import Mode


class DAG(object):
    def __init__(self, circuit: Circuit = None, mode: Mode = 1):
        """
        Params:
            circuit: The logical circuit.
            mode:  The mode how the code deal with circuit.
        """
        self._mode = mode
        if circuit is not None:
            if mode == Mode.WHOLE_CIRCUIT:
                self._transform_from_circuit(circuit=circuit)
            elif mode == Mode.TWO_QUBIT_CIRCUIT:
                self._construct_two_qubit_gates_circuit(circuit=circuit)
            else:
                raise Exception("The mode is not supported")
        else:
            self._dag = nx.MultiDiGraph()
        self._width = circuit.width()
        self._front_layer = None

    def __getitem__(self, index):
        """
        Return the gate node corresponding to the index  in the DAG
        """
        return self._dag.nodes[index]

    @property
    def width(self) -> int:
        """
        The number of qubits in the circuit
        """
        return self._width

    @property
    def size(self) -> int:
        """
        The number of nodes in DAG
        """
        return len(self._dag.nodes)

    @property
    def dag(self) -> nx.DiGraph:
        """
        The directed acyclic graph representation of the circuit
        """
        return self._dag

    @property
    def initial_qubit_mask(self) -> np.ndarray:
        """
        The qubit mask stores the index of current closet gate in the DAG, e.g.,
        given the  circuit with the sequential two-qubit gates {(0,1),(1,2),(2,3),(3,4),(4,1)},
        the qubit mask should be (0,0,1,2,3), which means the first and second physical
        should be allocated to the first gate for it's the first gate on the qubit wires and so on.

        Therefore, the initial_qubit_mask is the qubit mask of the initial whole circuit.
        """
        if self._mode == Mode.WHOLE_CIRCUIT:
            return None
        elif self._mode == Mode.TWO_QUBIT_CIRCUIT:
            return self._initial_qubit_mask
        else:
            raise Exception("The mode is not supported")

    @property
    def front_layer(self) -> List[int]:
        """
        The front layer of the DAG, which is the list of all the nodes with zero in-degree
        """
        if self._front_layer is not None:
            return self._front_layer.copy()
        else:
            front_layer = []
            for v in list(self._dag.nodes()):
                if self._dag.in_degree[v] == 0:
                    front_layer.append(v)
            self._front_layer = front_layer
            return self._front_layer.copy()

    @property
    def compact_dag(self) -> np.ndarray:
        """ """
        return np.concatenate([self._successors, self._precessors], axis=1)

    @property
    def node_qubits(self) -> np.ndarray:
        """ """
        return self._node_qubits

    @property
    def index(self) -> np.ndarray:
        """ """
        return self._index

    def get_successor_nodes(self, vertex: int) -> Iterable[int]:
        """
        Get the succeeding nodes of the current nodes
        """
        return self._dag.successors(vertex)

    def get_predecessor_nodes(self, vertex: int) -> Iterable[int]:
        """
        Get the preceeding nodes of the current nodes
        """
        return self._dag.predecessors(vertex)

    def get_egde_qubit(self, vertex_i: int, vertex_j: int):
        """
        The qubit associated with the edge from vertex i to vertex j
        """
        return self._dag.edges[vertex_i, vertex_j]["qubit"]

    def _transform_from_circuit(self, circuit: Circuit):
        """
        Transform the whole circuit into a directed acyclic graph
        """
        self._num_of_gate = 0
        self._qubit_mask = np.zeros(circuit.width(), dtype=np.int32) - 1
        self._initial_qubit_mask = np.zeros(circuit.width(), dtype=np.int32) - 1
        self._depth = np.zeros(circuit.size(), dtype=np.int32)

        self._dag = nx.MultiDiGraph()
        for gate in circuit.gates:
            self._dag.add_node(
                self._num_of_gate, gate=gate, depth=self._gate_depth(gate)
            )
            if gate.controls + gate.targets == 1:
                self._add_edge_in_dag(gate.targ)
            elif gate.controls + gate.targets == 2:
                qubits = tuple(gate.cargs + gate.targs)
                self._add_edge_in_dag(qubits[0])
                self._add_edge_in_dag(qubits[1])
            else:
                raise Exception(
                    str("The gate is not single qubit gate or two qubit gate")
                )
            self._num_of_gate = self._num_of_gate + 1

    def _construct_two_qubit_gates_circuit(self, circuit: Circuit):
        """
        Transform the sub circuit only with two-qubit gates  in the original circuit into a directed acyclic graph
        """
        self._num_of_gate = 0
        self._num_of_two_qubit_gate = 0

        self._qubit_mask = np.zeros(circuit.width(), dtype=np.int32) - 1
        self._initial_qubit_mask = np.zeros(circuit.width(), dtype=np.int32) - 1
        self._depth = np.zeros(circuit.size(), dtype=np.int32)
        self._dag = nx.DiGraph()

        # Compact representation of DAG
        self._successors = (
            np.zeros(shape=(circuit.count_2qubit_gate(), 2), dtype=np.int32) - 1
        )
        self._precessors = (
            np.zeros(shape=(circuit.count_2qubit_gate(), 2), dtype=np.int32) - 1
        )
        self._node_qubits = (
            np.zeros(shape=(circuit.count_2qubit_gate(), 2), dtype=np.int32) - 1
        )

        # Index and inverse index of gates from DAG to compact DAG
        self._index = np.zeros(circuit.size(), dtype=np.int32) - 1
        self._inverse_index = np.zeros(circuit.count_2qubit_gate(), dtype=np.int32) - 1

        for gate in circuit.gates:
            if gate.is_single():
                pass
            elif gate.controls + gate.targets == 2:
                if self._is_duplicate_gate(gate) is not True:
                    self._index[self._num_of_gate] = self._num_of_two_qubit_gate
                    self._inverse_index[self._num_of_two_qubit_gate] = self._num_of_gate
                    self._add_node_in_compact_dag(gate)
                    self._dag.add_node(self._num_of_gate, gate=gate, depth=self._gate_depth(gate))
                    qubits = gate.cargs + gate.targs
                    self._add_edge_in_dag(qubits[0])
                    self._add_edge_in_dag(qubits[1])

                    self._num_of_two_qubit_gate = self._num_of_two_qubit_gate + 1
            else:
                raise Exception(
                    str("The gate is not single qubit gate or two qubit gate")
                )

            self._num_of_gate = self._num_of_gate + 1

        for i in range(self._num_of_two_qubit_gate):
            self._fullfill_node(i)

    def _add_node_in_compact_dag(self, gate: BasicGate) -> bool:
        """
        Add the node's information, including its successors ,precessors and node's qubits, to the compact matrix
        """
        qubits = tuple(gate.cargs + gate.targs)

        precessors = [
            self._index[self._qubit_mask[qubits[0]]]
            if self._qubit_mask[qubits[0]] != -1
            else -1,
            self._index[self._qubit_mask[qubits[1]]]
            if self._qubit_mask[qubits[1]] != -1
            else -1,
        ]
        self._precessors[self._num_of_two_qubit_gate, :] = precessors

        self._add_successors(precessors[0], qubits[0])
        self._add_successors(precessors[1], qubits[1])

    def _add_successors(self, precessor: int, qubit: int):
        """
        Add the current two-qubit gate to its precessor as successor
        """
        if precessor != -1:
            if (
                self._successors[precessor][0] == -1
                and self._successors[precessor][1] == -1
            ):
                self._successors[precessor][0] = self._num_of_two_qubit_gate
                self._node_qubits[precessor][0] = qubit
            elif (
                self._successors[precessor][0] == -1
                and self._successors[precessor][1] != -1
            ):
                self._successors[precessor][0] = self._num_of_two_qubit_gate
                self._node_qubits[precessor][0] = qubit
            elif (
                self._successors[precessor][0] != -1
                and self._successors[precessor][1] == -1
            ):
                self._successors[precessor][1] = self._num_of_two_qubit_gate
                self._node_qubits[precessor][1] = qubit
            else:
                raise Exception("There is a conflict")

    def _fullfill_node(self, index: int):
        """
        Fullfill the node's information of qubits
        """
        if index != -1:
            gate = self._dag.nodes[self._inverse_index[index]]['gate']
            qubits = tuple(gate.cargs + gate.targs)

            i = index
            if self._node_qubits[i][0] == -1 and self._node_qubits[i][1] == -1:
                self._node_qubits[i] = np.array(qubits)
            elif self._node_qubits[i][0] == -1 and self._node_qubits[i][1] != -1:
                self._node_qubits[i][0] = (
                    qubits[0] if self._node_qubits[i][1] == qubits[1] else qubits[1]
                )
            elif self._node_qubits[i][0] != -1 and self._node_qubits[i][1] == -1:
                self._node_qubits[i][1] = (
                    qubits[1] if self._node_qubits[i][0] == qubits[0] else qubits[0]
                )
            else:
                pass

    def _is_duplicate_gate(self, gate: BasicGate) -> bool:
        """
        Indicate wether the gate share the same qubits with its preceeding gate
        """
        qubits = tuple(gate.cargs + gate.targs)

        if (
            self._qubit_mask[qubits[0]] != -1
            and self._qubit_mask[qubits[0]] == self._qubit_mask[qubits[1]]
        ):
            return True
        else:
            return False

    def _gate_depth(self, gate: BasicGate) -> int:
        """ """
        if gate.controls + gate.targets == 1:
            self._depth[self._num_of_gate] = self._gate_before_qubit_depth(gate.targ)

        elif gate.controls + gate.targets == 2:
            qubits = gate.cargs + gate.targs
            self._depth[self._num_of_gate] = max(self._gate_before_qubit_depth(qubits[0]),
                                                 self._gate_before_qubit_depth(qubits[1]))
        else:
            raise Exception(str("The gate is not single qubit gate or two qubit gate"))

        return self._depth[self._num_of_gate]

    def _gate_before_qubit_depth(self, qubit: int) -> int:
        """ """
        if self._qubit_mask[qubit] == -1:
            return 0
        else:
            return self._depth[self._qubit_mask[qubit]] + 1

    def _add_edge_in_dag(self, qubit: int):
        """ """
        if qubit < len(self._qubit_mask):
            if self._qubit_mask[qubit] != -1:
                self._dag.add_edge(
                    self._qubit_mask[qubit], self._num_of_gate, qubit=qubit
                )
            else:
                self._initial_qubit_mask[qubit] = self._num_of_gate
            self._qubit_mask[qubit] = self._num_of_gate
        else:
            raise Exception(str("   "))

    def get_subcircuit(
        self,
        front_layer: List[int],
        qubit_mask: List[int],
        num_of_gates: int,
        gates_threshold: int = -1,
    ) -> Tuple[int, np.ndarray]:
        """
        Get the subcircuit  from the front layer in the circuit
        """
        if gates_threshold == -1:
            gates_threshold = num_of_gates

        if gates_threshold < num_of_gates:
            raise Exception("The number of gates is exceed the threshold")

        mark = np.zeros(self.size + 1, dtype=np.int32) - 1
        # front_layer = [ self._index[idx]   for idx in front_layer]
        qubit_mask = [self._index[idx] if idx != -1 else -1 for idx in qubit_mask]
        # qubit_mask = np.zeros(self.size, dtype = np.int32) -1
        # for idx in front_layer:
        #     qubit_mask[self.node_qubits[idx][0]] = idx
        #     qubit_mask[self.node_qubits[idx][1]] = idx

        subcircuit = np.zeros(shape=(gates_threshold, 5), dtype=np.int32) - 1
        qubits_of_gates = np.zeros(shape=(gates_threshold, 2), dtype=np.int32) - 1

        front_layer_stack = deque([self._index[i] for i in front_layer])
        index = 0
        while index < num_of_gates and len(front_layer_stack) > 0:
            top = front_layer_stack.pop()
            mark[top] = index
            if index >= num_of_gates:
                print(list(front_layer_stack))
                for r in self._successors:
                    print(list(r))
                for p in self._precessors:
                    print(list(p))
                for s in subcircuit:
                    print(list(s))
                raise Exception(
                    "The number of gates in the subcircuit %d is greater than the given number %d"
                    % (index, num_of_gates)
                )
            subcircuit[index, 0] = top
            subcircuit[index, 1:] = np.concatenate(
                (self._successors[top], self._precessors[top])
            )
            qubits_of_gates[index, :] = self._node_qubits[top]
            index += 1
            for i, suc in enumerate(self._successors[top]):
                qubit_mask[self.node_qubits[top][i]] = suc
                if suc != -1 and mark[suc] == -1 and self._is_free(suc, qubit_mask):
                    # print(suc)
                    front_layer_stack.append(suc)
        subcircuit = mark[subcircuit]

        return index, np.concatenate([subcircuit, qubits_of_gates], axis=1)

    def _is_free(self, gate_idx: int, qubit_mask: List[int]):
        ctrl, tar = self.node_qubits[gate_idx]
        if (qubit_mask[ctrl] == -1 or qubit_mask[ctrl] == gate_idx) and (
            qubit_mask[tar] == -1 or qubit_mask[tar] == gate_idx
        ):
            return True
        else:
            return False

    def draw(self):
        """
        Draw the DAG of the circuit with
        """
        plt.figure(figsize=(200, 10))
        nx.draw(
            G=self._dag,
            pos=nx.multipartite_layout(self._dag, subset_key="depth"),
            node_size=50,
            width=1,
            arrowsize=2,
            font_size=12,
            with_labels=True,
        )
        # nx.draw(G = self._dag)
        plt.savefig("dag.png")
        plt.close()
