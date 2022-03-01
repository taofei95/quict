from random import randint
from typing import Iterator, Tuple, List
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
    TODO weak ref needed?
    DONE need to distinguish interfaces of a multi qubit gate
    """

    class Node:
        """
        DAG node class.
        """

        __slots__ = ['gate', 'predecessors', 'successors', 'flag', 'qubit_id', 'size']

        def __init__(self, gate_: BasicGate = None, qubit_=0):
            """
            Args:
                gate_(BasicGate): Gate represented by this node
                qubit_(int): the actual qubit the gate sits on (used only when `gate_` is None)
            """
            # TODO do we need to copy gate?
            self.gate = gate_
            self.qubit_id = {qubit_: i for i, qubit_ in gate_.affectArgs} if gate_ else {qubit_: 0}
            self.size = len(self.qubit_id)
            self.predecessors: List[Tuple[DAG.Node, int]] = [(None, 0)] * self.size
            self.successors: List[Tuple[DAG.Node, int]] = [(None, 0)] * self.size
            self.flag = 0

        def add_forward_edge(self, qubit_, node):
            u_id = self.qubit_id[qubit_]
            v_id = node.qubit_id[qubit_]
            self.successors[u_id] = (node, v_id)
            self.predecessors[v_id] = (self, v_id)

        def connect(self, forward_qubit, backward_qubit, node):
            self.successors[forward_qubit] = (node, backward_qubit)
            node.predecessors[backward_qubit] = (self, forward_qubit)

    def __init__(self, gates: CompositeGate):
        """
        Args:
            gates(CompositeGate): Circuit represented by this DAG
        """

        self.size = gates.circuit_width()
        self.start_nodes = [self.Node(qubit_=i) for i in range(self.size)]
        self.end_nodes = [self.Node(qubit_=i) for i in range(self.size)]
        self._build_graph(gates)

    def _build_graph(self, gates: CompositeGate):
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
            CompositeGate: Circuit equivalent to this DAG
        """

        circ = Circuit(self.size)
        mapping = {(id(node), 0): qubit_ for qubit_, node in enumerate(self.start_nodes)}
        for node in self.topological_sort():
            for qubit_ in range(node.size):
                pred, qubit2 = node.predecessors[qubit_]
                mapping[(id(node), qubit_)] = mapping[(id(pred), qubit2)]

            node.gate | circ([mapping[(id(node), qubit_)] for qubit_ in range(node.size)])
        return CompositeGate(circ)

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
