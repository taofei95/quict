from collections import defaultdict
from typing import Union, List, Dict


class Graph:
    @property
    def vector(self) -> int:
        return self._vectors

    @property
    def edges(self) -> dict:
        return self._edges

    def __init__(self, position: int, edges: Union[List, Dict] = None):
        self._vectors = position
        self._edges = defaultdict(list)
        if edges is not None:
            iterator = enumerate(edges) if isinstance(edges, list) else edges.items()
            for idx, edge in iterator:
                assert isinstance(edge, list)
                self._edges[idx] = edge

    def __str__(self):
        return str(self._edges)

    def add_edge(self, u: int, v: int):
        assert u >= 0 and u <= self._vectors
        assert v >= 0 and v <= self._vectors

        if v not in self._edges[u]:
            self._edges[u].append(v)

    def del_edge(self, u: int, v: int):
        assert u >= 0 and u <= self._vectors
        assert v >= 0 and v <= self._vectors

        if v in self._edges[u]:
            self._edges[u].remove(v)

    def validation(self) -> bool:
        edge_size = len(self._edges[0])
        for edge in self._edges.values():
            if len(edge) != edge_size:
                return False

        return True
