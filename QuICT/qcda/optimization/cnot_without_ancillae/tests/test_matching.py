import random

from .utility import get_bipartite, ensure_bipartite_max_degree_even
from ..edge_coloring import *


def test_matching():
    rnd = 50
    for _ in range(rnd):
        bipartite = get_bipartite(random.randint(100, 200))
        _, _, bipartite_regular = EdgeColoring.get_regular(bipartite)
        clr = 1
        match_cnt, colored_bipartite, stripped = EdgeColoring.get_colored_matching(
            bipartite_regular,
            clr,
            False
        )
        colored_cnt = 0
        nodes = []
        for node in colored_bipartite.left:
            eid = colored_bipartite.head[node]
            edge = colored_bipartite.edges[eid]
            s = edge.start
            e = edge.end
            c = edge.color
            assert c == clr
            nodes.append(s)
            nodes.append(e)
            colored_cnt += 1
        # It's really a perfect matching with color
        assert match_cnt == len(bipartite_regular.left)
        assert colored_cnt == match_cnt
        assert sorted(nodes) == sorted(bipartite_regular.nodes)
        # degree check for new graph
        for node in colored_bipartite.nodes:
            assert colored_bipartite.get_degree(node) == 1
        for node in stripped.nodes:
            assert stripped.get_degree(node) == bipartite_regular.get_degree(node) - 1
        # regularity check for original graph(to see if it changes)
        for node in bipartite_regular.nodes:
            assert bipartite_regular.get_degree(node) == bipartite_regular.get_any_degree()
