from collections import defaultdict
from typing import Dict, List, Union

import numpy as np


class Graph:
    """ The graph descript position space and action space for quantum random walk """

    @property
    def position(self) -> int:
        return self._nodes

    @property
    def position_qubits(self) -> int:
        return int(np.ceil(np.log2(self._nodes)))

    @property
    def action_space(self) -> int:
        return len(self._edges[0])

    @property
    def action_qubits(self) -> int:
        if self._operators is None:
            return int(np.ceil(np.log2(self.action_space)))
        else:
            shape = self._operators[0][0].shape
            return int(np.ceil(np.log2(shape[0])))

    @property
    def edges(self) -> dict:
        return self._edges

    def __init__(
        self, position: int, edges: Union[List, Dict] = None,
    ):
        """ Initial the random walk graph.

        Args:
            position (int): The number of graph's nodes.
            edges (Union[List, Dict], optional): The edges of each node. Defaults to None.
        """
        self._nodes = position
        self._edges = defaultdict(list)
        if edges is not None:
            iterator = enumerate(edges) if isinstance(edges, list) else edges.items()
            for idx, edge in iterator:
                assert isinstance(edge, list)
                self._edges[idx] = edge

    def __str__(self):
        return str(self._edges) + "\nOperators: " + str(self._operators)

    def add_edge(self, u: int, v: int):
        """ Add edge """
        assert u >= 0 and u <= self._nodes
        assert v >= 0 and v <= self._nodes

        if v not in self._edges[u]:
            self._edges[u].append(v)

    def del_edge(self, u: int, v: int):
        """ Remove edge """
        assert u >= 0 and u <= self._nodes
        assert v >= 0 and v <= self._nodes

        if v in self._edges[u]:
            self._edges[u].remove(v)

    def validation(self) -> bool:
        """ Validate that all vectors has same number of edges. """
        edge_size = len(self._edges[0])
        for edge in self._edges.values():
            if len(edge) != edge_size:
                return False

        return True
