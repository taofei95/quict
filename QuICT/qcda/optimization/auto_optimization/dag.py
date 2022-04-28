from random import randint
from typing import Iterator, Tuple, List, Set, Dict
from collections import Iterable, deque
import inspect

import numpy as np
from QuICT.core import *
from QuICT.core.gate import *


class DAG(Iterable):
    """
    DAG representation of a quantum circuit that indicates the commutative
    relations between gates. Iterate over a DAG will gate a sequence of
    BasicGate's in topological order.

    DONE converter between netlist and DAG
    DONE topological sort
    DONE sub circuit enumeration
    TODO weak ref needed
    DONE need to distinguish interfaces of a multi qubit gate
    TODO refactor flag system
    """

    class Node:
        """
        DAG node class.
        """

        __slots__ = ['gate', 'predecessors', 'successors', 'flag', 'qubit_id', 'size', 'qubit_loc']

        FLAG_DEFAULT = 0
        FLAG_VISITED = 1
        FLAG_ERASED = -1
        FLAG_IN_QUE = 2
        # FLAG_ANCHOR = 3

        def __init__(self, gate_: BasicGate = None, qubit_=0):
            """
            Args:
                gate_(BasicGate): Gate represented by this node
                qubit_(int): the actual qubit the gate sits on (used only when `gate_` is None)
            """
            self.gate = None if gate_ is None else gate_.copy()
            # the actual qubit the i-th wire of the gate act on
            self.qubit_loc = [qubit_] if gate_ is None else list(gate_.affectArgs)
            # inverse mapping of self.qubit_loc
            self.qubit_id = {qubit_: 0} if gate_ is None else {qubit_: i for i, qubit_ in enumerate(self.qubit_loc)}
            self.size = len(self.qubit_id)
            self.predecessors: List[Tuple[DAG.Node, int]] = [(None, 0)] * self.size
            self.successors: List[Tuple[DAG.Node, int]] = [(None, 0)] * self.size
            self.flag = self.FLAG_DEFAULT
            self.size = len(self.qubit_id)

        def add_forward_edge(self, qubit_, node):
            """
            Add an edge from self to node. Make sure qubit_id is up to date.

            Args:
                qubit_(int): the actual qubit this edge sit on
                node(DAG.Node): the node this edge point to
            """
            u_id = self.qubit_id[qubit_]
            v_id = node.qubit_id[qubit_]
            # print(u_id, v_id, self.size)
            self.successors[u_id] = (node, v_id)
            node.predecessors[v_id] = (self, u_id)

        def connect(self, forward_qubit, backward_qubit, node):
            """
            Connect (self, forward_qubit) to (node, backward_qubit)

            Args:
                forward_qubit(int): the wire this edge points from
                backward_qubit(int): the wire this edge points to
                node(DAG.Node):  the node this edge points to
            """
            self.successors[forward_qubit] = (node, backward_qubit)
            node.predecessors[backward_qubit] = (self, forward_qubit)

        def erase(self):
            """
            erase this node from the DAG. It will:
            1. set this node as FLAG_ERASED
            2. remove pointers between it and adjacent nodes
            """
            self.flag = self.FLAG_ERASED
            for qubit_ in range(self.size):
                p_node, p_qubit = self.predecessors[qubit_]
                n_node, n_qubit = self.successors[qubit_]
                if p_node:
                    p_node.connect(p_qubit, n_qubit, n_node)
                elif n_node:
                    n_node.successors[qubit_] = (None, 0)
            # FIXME: the following assignment causes runtime error
            # self.predecessors = None
            # self.successors = None

    def __init__(self, gates: Circuit):
        """
        Args:
            gates(Circuit): Circuit represented by this DAG
        """

        self.size = gates.circuit_width()
        self.start_nodes = [self.Node(qubit_=i) for i in range(self.size)]
        self.end_nodes = [self.Node(qubit_=i) for i in range(self.size)]
        self.global_phase = 0
        self._build_graph(gates)

    def circuit_width(self):
        """
        Get number of qubits of this circuit
        Returns:
            int: Number of qubits
        """
        return self.size

    def _build_graph(self, gates: Circuit):
        cur_nodes = self.start_nodes.copy()
        for gate_ in gates.gates:
            gate_: BasicGate
            if gate_.qasm_name in ['ccz', 'ccx']:
                raise Exception('ccz / ccx gates not supported yet' )

            node = self.Node(gate_)
            for qubit_ in gate_.affectArgs:
                cur_nodes[qubit_].add_forward_edge(qubit_, node)
                cur_nodes[qubit_] = node

        for qubit_ in range(self.size):
            cur_nodes[qubit_].add_forward_edge(qubit_, self.end_nodes[qubit_])

    def get_circuit(self):
        """
        Generate circuit net list from this DAG.

        Returns:
            Circuit: Circuit equivalent to this DAG
        """

        circ = Circuit(self.size)
        mapping = {(id(node), 0): qubit_ for qubit_, node in enumerate(self.start_nodes)}
        for node in self.topological_sort():
            # print(node.gate.qasm_name)
            for qubit_ in range(node.size):
                pred, qubit2 = node.predecessors[qubit_]
                mapping[(id(node), qubit_)] = mapping[(id(pred), qubit2)]

            node.gate | circ([mapping[(id(node), qubit_)] for qubit_ in range(node.size)])

        if not np.isclose(self.global_phase, 0):
            Phase(self.global_phase) | circ(0)
        return circ

    def topological_sort(self, include_dummy=False):
        """
        Iterate over nodes in this DAG in topological order (ignore start nodes)

        Returns:
            Iterator[DAG.Node]: gates in topological order
        """

        edge_count = {}
        queue = deque(self.start_nodes)
        while len(queue) > 0:
            cur = queue.popleft()
            if cur.gate is not None or include_dummy:
                yield cur
            for nxt, _ in cur.successors:
                if nxt is None:
                    continue
                if id(nxt) not in edge_count:
                    edge_count[id(nxt)] = nxt.size
                edge_count[id(nxt)] -= 1
                if edge_count[id(nxt)] == 0:
                    queue.append(nxt)

    @staticmethod
    def topological_sort_sub_circuit(prev_node, succ_node, include_dummy=False):
        # TODO review edge cases that one income node is not in prev_node
        end_set = set()
        for each in succ_node:
            if each is None:
                continue
            node_, qubit_ = each
            end_set.add((id(node_), qubit_))

        edge_count = {}
        queue = deque()
        for each in prev_node:
            if each is None:
                continue
            node_, qubit_ = each
            nxt, nxt_q = node_.successors[qubit_]
            if (id(nxt), nxt_q) in end_set:
                continue

            if id(nxt) not in edge_count:
                edge_count[id(nxt)] = nxt.size
            edge_count[id(nxt)] -= 1
            if edge_count[id(nxt)] == 0:
                queue.append(nxt)

        while len(queue) > 0:
            cur = queue.popleft()
            if cur.gate is not None or include_dummy:
                yield cur
            for nxt, nxt_q in cur.successors:
                if nxt is None or (id(nxt), nxt_q) in end_set:
                    continue
                if id(nxt) not in edge_count:
                    edge_count[id(nxt)] = nxt.size
                edge_count[id(nxt)] -= 1
                if edge_count[id(nxt)] == 0:
                    queue.append(nxt)

    def reset_flag(self):
        for node in self.start_nodes:
            node.flag = DAG.Node.FLAG_DEFAULT
        for node in self.end_nodes:
            node.flag = DAG.Node.FLAG_DEFAULT
        for node in self.topological_sort():
            node.flag = DAG.Node.FLAG_DEFAULT

    def set_qubit_loc(self):
        mapping = {(id(node), 0): qubit_ for qubit_, node in enumerate(self.start_nodes)}
        for i, node in enumerate(self.start_nodes):
            node.qubit_loc[0] = i
            node.qubit_id = {i: 0}
        for i, node in enumerate(self.end_nodes):
            node.qubit_loc[0] = i
            node.qubit_id = {i: 0}

        for node in self.topological_sort():
            for qubit_ in range(node.size):
                pred, qubit2 = node.predecessors[qubit_]
                mapping[(id(node), qubit_)] = mapping[(id(pred), qubit2)]
                node.qubit_loc[qubit_] = mapping[(id(node), qubit_)]
            if node.gate is not None:
                node.gate.affectArgs = node.qubit_loc
            node.qubit_id = {qubit_: i for i, qubit_ in enumerate(node.qubit_loc)}

    def compare_circuit(self, other: Tuple[Node, int], anchor_qubit: int, flag_enabled: bool = False):

        t_node, t_qubit = self.start_nodes[anchor_qubit].successors[0]
        o_node, o_qubit = other[0], (t_qubit if other[1] == -1 else other[1])
        if o_node.gate is None:
            return None

        if o_node.gate.qasm_name != t_node.gate.qasm_name or (t_qubit != o_qubit) or (o_node.flag and flag_enabled):
            return None

        mapping = {id(t_node): o_node}
        queue = deque([(t_node, o_node)])
        while len(queue) > 0:
            u, v = queue.popleft()
            for neighbors in ['predecessors', 'successors']:
                for qubit_ in range(u.size):
                    u_nxt, u_qubit = getattr(u, neighbors)[qubit_]
                    assert u_nxt, "u_nxt == None should not happen"
                    if not u_nxt.gate:
                        continue

                    v_nxt, v_qubit = getattr(v, neighbors)[qubit_]
                    if id(u_nxt) in mapping:
                        if id(mapping[id(u_nxt)]) != id(v_nxt) or u_qubit != v_qubit:
                            return None
                        continue

                    assert v_nxt, "v_nxt == None should not happen"
                    if not v_nxt.gate or u_qubit != v_qubit or \
                            (v_nxt.flag and flag_enabled) or \
                            u_nxt.gate.qasm_name != v_nxt.gate.qasm_name:
                        return None

                    mapping[id(u_nxt)] = v_nxt
                    queue.append((u_nxt, v_nxt))

        if flag_enabled:
            for each in mapping.values():
                each.flag = DAG.Node.FLAG_VISITED
        return mapping

    @staticmethod
    def replace_circuit(mapping: Dict[int, Tuple[Node, int]], replacement):
        replacement: DAG
        for qubit_ in range(replacement.size):
            # first node on qubit_ in replacement circuit
            r_node, r_qubit = replacement.start_nodes[qubit_].successors[0]
            # node previous to the node corresponding to t_node in original circuit
            if id(replacement.start_nodes[qubit_]) not in mapping:
                # TODO can we remove this if
                continue
            p_node, p_qubit = mapping[id(replacement.start_nodes[qubit_])]
            # place r_node after p_node
            p_node.connect(p_qubit, r_qubit, r_node)

        for qubit_ in range(replacement.size):
            # last node on qubit_ in replacement circuit
            r_node, r_qubit = replacement.end_nodes[qubit_].predecessors[0]
            # node successive to the node corresponding to t_node in original circuit
            if id(replacement.end_nodes[qubit_]) not in mapping:
                continue
            s_node, s_qubit = mapping[id(replacement.end_nodes[qubit_])]
            # place s_node after r_node
            r_node.connect(r_qubit, s_qubit, s_node)

    @staticmethod
    def _get_reachable_relation(node: Node, qubit_: int) -> Set[Tuple[Tuple[int, int], int]]:
        visited = set()
        queue = deque([(node, qubit_)])
        while len(queue) > 0:
            cur, cur_q = queue.popleft()
            if cur.gate is None:
                continue
            nxt, _ = cur.successors[cur_q]
            if id(nxt) not in visited:
                for nxt_q in range(nxt.size):
                    queue.append((nxt, nxt_q))
                visited.add(id(nxt))

        reachable = {((id(node), qubit_), o) for o in visited}
        return reachable

    def get_reachable_relation(self) -> Set[Tuple[Tuple[int, int], int]]:
        reachable = set()
        for each in self.topological_sort(include_dummy=True):
            for qubit_ in range(each.size):
                reachable.update(self._get_reachable_relation(each, qubit_))
        return reachable

    @staticmethod
    def copy_sub_circuit(prev_node: List[Tuple[Node, int]], succ_node: List[Tuple[Node, int]]):
        circ = Circuit(len(prev_node))
        for node in DAG.topological_sort_sub_circuit(prev_node, succ_node):
            node.gate.copy() | circ(node.qubit_loc)
        return DAG(circ)

    def __iter__(self):
        """
        Iterate over gates in this DAG in topological order

        Returns:
            Iterator[BasicGate]: gates in topological order
        """

        for node in self.topological_sort():
            yield node.gate

    def copy(self):
        # TODO faster implementation of copy()
        return DAG(self.get_circuit())

