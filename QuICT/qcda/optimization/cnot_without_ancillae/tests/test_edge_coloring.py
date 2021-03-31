import random

from .utility import get_bipartite, ensure_bipartite_max_degree_even
from ..edge_coloring import *


def test_edge_coloring_for_regular():
    rnd = 20
    for _ in range(rnd):
        bipartite = get_bipartite(100)
        _, _, bipartite_regular = EdgeColoring.get_regular(bipartite)
        colored = EdgeColoring.get_edge_coloring_for_regular(bipartite_regular, 1)
        deg = colored.get_any_degree()
        # test regularity
        for node in colored.nodes:
            assert colored.get_degree(node) == deg
        # test color
        for node in colored.nodes:
            clr_set = set()
            eid = colored.head[node]
            while eid != -1:
                edge = colored.edges[eid]
                clr = edge.color
                assert clr != -1
                assert clr not in clr_set
                clr_set.add(clr)
                eid = edge.next
            assert deg == len(clr_set)


def test_edge_coloring():
    rnd = 20
    for _ in range(rnd):
        bipartite = get_bipartite(100)
        colored_bipartite = EdgeColoring.get_edge_coloring(bipartite)
        # check color
        for node in colored_bipartite.nodes:
            eid = colored_bipartite.head[node]
            color_set = set()
            while eid != -1:
                edge = colored_bipartite.edges[eid]
                # color_set
                assert edge.color not in color_set
                color_set.add(edge.color)
                eid = edge.next
        # check id of nodes
        assert sorted(bipartite.left) == sorted(colored_bipartite.left)
        assert sorted(bipartite.right) == sorted(colored_bipartite.right)
        # check edges
        for node in colored_bipartite.left:
            eid = colored_bipartite.head[node]
            other_side_new = []
            other_side = []
            while eid != -1:
                edge = colored_bipartite.edges[eid]
                other_side_new.append(edge.end)
                eid = edge.next
            eid = bipartite.head[node]
            while eid != -1:
                edge = bipartite.edges[eid]
                other_side.append(edge.end)
                eid = edge.next
            assert sorted(other_side) == sorted(other_side_new)
