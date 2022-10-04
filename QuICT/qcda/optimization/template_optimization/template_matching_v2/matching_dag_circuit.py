from typing import Set, List
from collections.abc import Iterable

from QuICT.core import Circuit
from QuICT.core.circuit.dag_circuit import DAGNode, DAGCircuit
from QuICT.core.gate import BasicGate
from collections import deque
from collections import namedtuple


NodeInfo = namedtuple('NodeInfo', ['matched_with', 'is_blocked'])


class MatchingDAGNode(DAGNode):
    __slots__ = ['successors_to_visit', 'matched_with', 'is_blocked',
                 '_id', '_gate', '_name', '_cargs', '_targs', '_qargs',
                 '_successors', '_predecessors']

    def __init__(self, id: int, gate: BasicGate):
        self.successors_to_visit = []
        self.matched_with = None
        self.is_blocked = False
        super().__init__(id, gate)

    def pop_successors_to_visit(self):
        if not self.successors_to_visit:
            return None

        ret = self.successors_to_visit[0]
        self.successors_to_visit.pop(0)
        return ret

    def matchable(self):
        return (not self.is_blocked) and (self.matched_with is None)

    def compare_with(self, other, qubit_mapping=None) -> bool:
        if self.name != other.name:
            return False
        if qubit_mapping is not None:
            for t_qubit, c_qubit in zip(self.qargs, other.qargs):
                if qubit_mapping[t_qubit] != c_qubit:
                    return False
        return True

    def node_info(self):
        return NodeInfo(self.matched_with, self.is_blocked)


class MatchingDAGCircuit(DAGCircuit):
    def _to_dag_circuit(self):
        gates = self._circuit.gates
        endpoints = []      # The endpoint of current DAG graph
        for idx, gate in enumerate(gates):
            # Add new node into DAG Graph
            assert isinstance(gate, BasicGate), "Only support BasicGate in DAGCircuit."
            current_node = MatchingDAGNode(idx, gate)
            self.add_node(current_node)

            # Check the relationship of current node and previous node
            updated_endpoints = []
            for previous_node in endpoints:
                self._backward_trace(previous_node, current_node)
                if self._graph.edges(idx) != 0:
                    updated_endpoints.append(current_node)
                    if not self._graph.has_edge(previous_node.id, idx):
                        updated_endpoints.append(previous_node)
                else:
                    updated_endpoints.append(previous_node)

            # if no edges add, create new original node
            if self._graph.degree(idx) == 0:
                endpoints.insert(0, current_node)
            else:
                endpoints = self._endpoints_order(updated_endpoints)

        # Add successors and predecessors for all nodes
        for node_id in range(self.size):
            node_sces = self._graph.successors(node_id)
            node_pdces = self._graph.predecessors(node_id)
            self.get_node(node_id).successors = list(node_sces)
            self.get_node(node_id).predecessors = list(node_pdces)

    def init_forward_matching(self, node_id, other_id, s2v_enabled=False):
        for nid in self.nodes():
            node = self.get_node(nid)
            if node.id == node_id:
                node.matched_with = other_id
                if s2v_enabled:
                    node.successors_to_visit = sorted(node.successors.copy())
            else:
                node.matched_with = None
                if s2v_enabled:
                    node.successors_to_visit = []
            node.is_blocked = False

    def _all_reachable(self, start, direction):
        if isinstance(start, int):
            visited = {start}
        elif isinstance(start, Iterable):
            visited = set(start)
        else:
            assert False, 'start must be int or iterable objects'

        que = deque(visited)
        init_visited = visited.copy()
        while len(que) > 0:
            cur_node = self.get_node(que.popleft())
            for node_id in getattr(cur_node, direction):
                if node_id not in visited:
                    que.append(node_id)
                    visited.add(node_id)

        return visited - init_visited

    def all_predecessors(self, start) -> Set[int]:
        return self._all_reachable(start, 'predecessors')

    def all_successors(self, start) -> Set[int]:
        return self._all_reachable(start, 'successors')

    def matching_info(self):
        return [self.get_node(i).node_info() for i in range(self.size)]

    def get_circuit(self):
        circ = Circuit(self.width)
        for node_id in self.nodes():
            node: MatchingDAGNode = self.get_node(node_id)
            circ.append(node.gate.copy())
        return circ


class Match:
    """
    Class to store matches
    """
    def __init__(self, match: List, qubit_mapping: List):
        self.match = sorted(match)
        self.qubit_mapping = qubit_mapping
        self._template_nodes = None
        self._circuit_nodes = None

    def __len__(self):
        return len(self.match)

    def __hash__(self):
        return hash(tuple(self.match))

    def __str__(self):
        return f'Match({str(self.match)}, {str(self.qubit_mapping)})'

    def __repr__(self):
        return str(self)

    @property
    def template_nodes(self):
        if self._template_nodes is None:
            self._template_nodes = {m[0] for m in self.match}
        return self._template_nodes

    @property
    def circuit_nodes(self):
        if self._circuit_nodes is None:
            self._circuit_nodes = {m[1] for m in self.match}
        return self._circuit_nodes
