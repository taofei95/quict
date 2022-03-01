from random import randint
from typing import Iterator
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
    TODO sub circuit enumeration
    TODO weak ref needed
    TODO need to distinguish interfaces of a multi qubit gate
    """

    class Node:
        """
        DAG node class.
        """
        __slots__ = ['gate', 'predecessors', 'successors', 'flag']

        def __init__(self, gate_: BasicGate = None):
            """
            Args:
                gate_(BasicGate): Gate represented by this node
            """
            self.gate = gate_
            self.predecessors = {}
            self.successors = {}
            self.flag = 0

        def __del__(self):
            """
            For debug use
            """
            print(f'in {inspect.currentframe()}: {self.gate.qasm_name} deconstructed')

        def check_equivalence(self, other, flag=0):
            """
            Check if the whole circuit starting from `self` equals a partial circuit starting
            from `other` (only consider gate types)

            Args:
                other(Node): The node in another DAG corresponding to self
                flag(int): Set the field `flag` of nodes in the other circuit to this value if check successful

            Returns:
                dict[int, DAG.Node]: Return None if not equal. Otherwise return node-to-node
                mapping from this circuit's nodes (id) to the other circuit's nodes
            """

            if other.gate.qasm_name != self.gate.qasm_name:
                return False
            visited_node = {id(self): other}
            queue = deque([(self, other)])
            while len(queue) > 0:
                u, v = queue.popleft()
                for neighbors in ['predecessors', 'successors']:
                    for qu, qv in zip(u.gate.affectArgs, v.gate.affectArgs):
                        if qu in getattr(u, neighbors):
                            u_nxt = getattr(u, neighbors)[qu]
                            if u_nxt.gate and id(u_nxt) not in visited_node:
                                if qv not in getattr(v, neighbors):
                                    return None
                                v_nxt = getattr(v, neighbors)[qv]
                                if u_nxt.gate.qasm_name != v_nxt.gate.qasm_name:
                                    return None
                                visited_node[id(u_nxt)] = v_nxt
                                queue.append((u_nxt, v_nxt))
            for v in visited_node.values():
                v.flag = flag
            return visited_node

    def __init__(self, gates: CompositeGate):
        """
        Args:
            gates(CompositeGate): Circuit represented by this DAG
        """

        # TODO how to get number of qubits
        self._start_nodes = [self.Node() for _ in range(gates.circuit_width())]
        self._build_graph(gates)

    def _build_graph(self, gates: CompositeGate):
        for gate_ in gates:
            gate_: BasicGate
            node = self.Node(gate_)
            cur_nodes = self._start_nodes.copy()

            for qubit_ in gate_.affectArgs:
                cur_nodes[qubit_].successors[qubit_] = node
                node.predecessors[qubit_] = cur_nodes[qubit_]
                cur_nodes[qubit_] = node

    def get_circuit(self):
        """
        Generate circuit net list from this DAG.

        Returns:
            CompositeGate: Circuit equivalent to this DAG
        """
        return CompositeGate(list(self))

    def topological_sort(self):
        """
        Iterate over nodes in this DAG in topological order (ignore start nodes)

        Returns:
            Iterator[DAG.Node]: gates in topological order
        """
        edge_count = {}
        queue = deque(self._start_nodes)
        while len(queue) > 0:
            cur = queue.popleft()
            if cur.gate:
                yield cur

            for nxt in cur.successors:
                if id(nxt) not in edge_count:
                    edge_count[id(nxt)] = len(nxt.predecessors)
                edge_count[id(nxt)] -= 1
                if edge_count[id(nxt)] == 0:
                    queue.append(nxt)

    def match_sub_circuit(self, pattern_node):
        """
        Iterate over every non-overlap sub-circuit in this DAG that are equivalent to the pattern.
        Do not modify the DAG during the iteration.

        TODO complete docstring
        """
        rand_flag = randint(0, 0xffffffff)
        for node in self.topological_sort():
            if node.flag != rand_flag:
                visited_node = pattern_node.check_equivalence(node, flag=rand_flag)
                if visited_node:
                    yield visited_node

    def replace_sub_circuit(self):
        pass

    def __iter__(self):
        """
        Iterate over gates in this DAG in topological order

        Returns:
            Iterator[BasicGate]: gates in topological order
        """
        for node in self.topological_sort():
            yield node.gate
