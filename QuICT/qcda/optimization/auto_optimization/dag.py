from random import randint
from typing import Iterator, Tuple, List, Set, Dict
from collections import Iterable, deque
import inspect

from QuICT.core import *


class DAG(Iterable):
    """
    DAG representation of a quantum circuit that indicates the commutative
    relations between gates. Iterate over a DAG will gate a sequence of
    BasicGate's in topological order.

    DONE converter between netlist and DAG
    DONE topological sort
    DONE sub circuit enumeration
    TODO weak ref needed!
    DONE need to distinguish interfaces of a multi qubit gate
    TODO should null node be set visited?
    """

    class Node:
        """
        DAG node class.
        """

        __slots__ = ['gate', 'predecessors', 'successors', 'flag', 'qubit_id', 'size', 'qubit_loc']

        FLAG_DEFAULT = 0
        FLAG_VISITED = 1
        FLAG_ERASED = -1

        def __init__(self, gate_: BasicGate = None, qubit_=0):
            """
            Args:
                gate_(BasicGate): Gate represented by this node
                qubit_(int): the actual qubit the gate sits on (used only when `gate_` is None)
            """
            # TODO do we need to copy gate?
            self.gate = gate_.copy()
            self.qubit_id = {qubit_: i for i, qubit_ in enumerate(gate_.affectArgs)} if gate_ else {qubit_: 0}
            self.size = len(self.qubit_id)
            self.predecessors: List[Tuple[DAG.Node, int]] = [(None, 0)] * self.size
            self.successors: List[Tuple[DAG.Node, int]] = [(None, 0)] * self.size
            self.flag = self.FLAG_DEFAULT
            self.size = len(self.qubit_id)

        def add_forward_edge(self, qubit_, node):
            u_id = self.qubit_id[qubit_]
            v_id = node.qubit_id[qubit_]
            self.successors[u_id] = (node, v_id)
            self.predecessors[v_id] = (self, v_id)

        def connect(self, forward_qubit, backward_qubit, node):
            self.successors[forward_qubit] = (node, backward_qubit)
            node.predecessors[backward_qubit] = (self, forward_qubit)

        def erase(self):
            self.flag = self.FLAG_ERASED
            for qubit_ in range(self.size):
                p_node, p_qubit = self.predecessors[qubit_]
                n_node, n_qubit = self.successors[qubit_]
                if p_node:
                    p_node.connect(p_qubit, n_qubit, n_node)
                elif n_node:
                    n_node.successors[qubit_] = (None, 0)

    def __init__(self, gates: Circuit):
        """
        Args:
            gates(Circuit): Circuit represented by this DAG
        """

        self.size = gates.circuit_width()
        self.start_nodes = [self.Node(qubit_=i) for i in range(self.size)]
        self.end_nodes = [self.Node(qubit_=i) for i in range(self.size)]
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
        for gate_ in gates:
            gate_: BasicGate
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
            for qubit_ in range(node.size):
                pred, qubit2 = node.predecessors[qubit_]
                mapping[(id(node), qubit_)] = mapping[(id(pred), qubit2)]

            node.gate | circ([mapping[(id(node), qubit_)] for qubit_ in range(node.size)])
        return Circuit(circ)

    def topological_sort(self):
        """
        Iterate over nodes in this DAG in topological order (ignore start nodes)

        Returns:
            Iterator[DAG.Node]: gates in topological order
        """

        edge_count = {}
        queue = deque(self.start_nodes)
        while len(queue) > 0:
            cur = queue.popleft()
            if cur.gate:
                yield cur
            for nxt, _ in cur.successors:
                if nxt is None:
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
        for i, node in enumerate(self.end_nodes):
            node.qubit_loc[0] = i

        for node in self.topological_sort():
            for qubit_ in range(node.size):
                pred, qubit2 = node.predecessors[qubit_]
                mapping[(id(node), qubit_)] = mapping[(id(pred), qubit2)]
                node.qubit_loc[qubit_] = mapping[(id(node), qubit_)]

    @staticmethod
    def _search_sub_circuit(start_node: Node, gate_set: Set[str], n_qubit: int):
        prev_node: List[Tuple[DAG.Node, int]] = [None] * n_qubit
        succ_node: List[Tuple[DAG.Node, int]] = [None] * n_qubit

        start_node.flag = DAG.Node.FLAG_VISITED
        queue = deque([start_node])
        while len(queue) > 0:
            cur = queue.popleft()
            for node_, qubit_ in cur.predecessors:
                if node_.gate.qasm_name in gate_set and all([prev_node[k] is None for k in node_.qubit_loc]):
                    queue.append(node_)
                    node_.flag = DAG.Node.FLAG_VISITED
                else:
                    prev_node[qubit_] = (node_, qubit_)
            for node_, qubit_ in cur.successors:
                if node_.gate.qasm_name in gate_set and all([succ_node[k] is None for k in node_.qubit_loc]):
                    queue.append(node_)
                    node_.flag = DAG.Node.FLAG_VISITED
                else:
                    succ_node[qubit_] = (node_, qubit_)
        return prev_node, succ_node

    def enumerate_sub_circuit(self, gate_set: Set[str]):
        self.reset_flag()
        for node_ in self.topological_sort():
            if node_.flag != node_.FLAG_VISITED and node_.gate.qasm_name in gate_set:
                yield self._search_sub_circuit(node_, gate_set, self.size)

    def compare_circuit(self, other: Node, anchor_qubit: int, flag_enabled: bool = False):
        o_node, o_qubit = other
        t_node, t_qubit = self.start_nodes[anchor_qubit].successors[0]
        if o_node.gate.qasm_name != t_node.gate.qasm_name or o_qubit != t_qubit or \
                (o_node.flag and flag_enabled):
            return None

        mapping = {id(t_node): o_node}
        queue = deque([(t_node, o_node)])
        while len(queue) > 0:
            u, v = queue.popleft()
            for neighbors in ['predecessors', 'successors']:
                for qubit_ in range(u.size):
                    u_nxt, u_qubit = getattr(u, neighbors)[qubit_]
                    assert u_nxt, "u_nxt == None should not happen"
                    if not u_nxt.gate or id(u_nxt) in mapping:
                        continue

                    v_nxt, v_qubit = getattr(v, neighbors)[qubit_]
                    assert v_nxt, "v_nxt == None should not happen"
                    if not v_nxt.gate or u_qubit != v_qubit or \
                            (v_nxt.flag and flag_enabled) or \
                            u_nxt.gate.qasm_name != v_nxt.gate.qasm_name:
                        return None

                    mapping[id(u_nxt)] = v_nxt
                    queue.append((id(u_nxt), v_nxt))

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
    def create_sub_circuit(prev_node: List[Tuple[Node, int]], succ_node: List[Tuple[Node, int]]):
        circ = Circuit(len(prev_node))
        dag = DAG(circ)
        for qubit_ in range(dag.size):
            if not prev_node[qubit_] or not succ_node[qubit_]:
                continue

            p_node, p_qubit = prev_node[qubit_]
            n_node, n_qubit = p_node.successors[p_qubit]
            dag.start_nodes[qubit_].connect(0, n_qubit, n_node)

            p_node, p_qubit = succ_node[qubit_]
            n_node, n_qubit = p_node.predecessors[p_qubit]
            n_node.connect(n_qubit, 0, dag.end_nodes[qubit_])
        return dag

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

