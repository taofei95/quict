from collections import Iterable

from QuICT.core import *


class DAG(Iterable):
    """
    DAG representation of a quantum circuit that indicates the commutative
    relations between gates. Iterate over a DAG will gate a sequence of
    BasicGate's in topological order.

    DONE converter between netlist and DAG
    DONE iterator of neighbors
    TODO pattern searching
    TODO sub circuit enumeration
    """

    class Node:
        """
        DAG node class.
        """
        __slots__ = ['gate', 'visited', 'predecessors', 'successors']

        def __init__(self, gate_: BasicGate = None):
            """
            Args:
                gate_(BasicGate): Gate represented by this node
            """
            self.gate = gate_
            self.visited = False
            self.predecessors = {}
            self.successors = {}

    def __init__(self, gates: CompositeGate):
        """
        Args:
            gates(CompositeGate): Circuit represented by this DAG
        """
        self.nodes = []
        # DONE circuit_width should be replaced
        self._start_nodes = {}
        self._end_nodes = {}
        self._build_graph(gates)

    def _build_graph(self, gates: CompositeGate):
        for gate_ in gates:
            gate_: BasicGate
            node = self.Node(gate_)
            self.nodes.append(node)

            for qubit_ in gate_.affectArgs:
                if qubit_ not in self._start_nodes:
                    self._start_nodes[qubit_] = self._end_nodes[qubit_] = node
                else:
                    self._end_nodes[qubit_].successors[qubit_] = node
                    node.predecessors[qubit_] = self._end_nodes[qubit_]
                    self._end_nodes[qubit_] = node

    def get_circuit(self):
        """
        Generate circuit net list from this DAG.

        Returns:
            CompositeGate: Circuit equivalent to this DAG
        """
        return CompositeGate([node.gate for node in self.nodes])

    def __iter__(self):
        for node in self.nodes:
            yield node.gate
