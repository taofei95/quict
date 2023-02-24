import random

from .utility import get_bipartite
from QuICT.qcda.optimization.cnot_without_ancilla import *


def test_matching_keep_origin():
    change_origin = False
    bipartite = get_bipartite(random.randint(100, 200), False)
    _, _, bipartite_regular = EdgeColoring.get_regular(bipartite)
    clr = 1
    match_cnt, colored_bipartite, stripped = EdgeColoring.get_colored_matching(
        bipartite_regular,
        clr,
        change_origin
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
    # Degree check for new graph
    for node in colored_bipartite.nodes:
        assert colored_bipartite.get_degree(node) == 1
    for node in stripped.nodes:
        assert stripped.get_degree(node) == bipartite_regular.get_degree(node) - 1
    # Regularity check for original graph(to see if it changes)
    for node in bipartite_regular.nodes:
        assert bipartite_regular.get_degree(node) == bipartite_regular.get_any_degree()
    if not change_origin:
        # Check color does not change in original graph(default color is -1).
        for edge in bipartite_regular.edges:
            assert edge.color != clr
    else:
        origin_edge_clr_cnt = 0
        for edge in bipartite_regular.edges:
            if edge.color == clr:
                origin_edge_clr_cnt += 1
        origin_edge_clr_cnt //= 2
        assert origin_edge_clr_cnt == match_cnt


def test_matching_color_origin():
    change_origin = True
    bipartite = get_bipartite(random.randint(100, 200), False)
    _, _, bipartite_regular = EdgeColoring.get_regular(bipartite)
    clr = 1
    match_cnt, colored_bipartite, stripped = EdgeColoring.get_colored_matching(
        bipartite_regular,
        clr,
        change_origin
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
    # Degree check for new graph
    for node in colored_bipartite.nodes:
        assert colored_bipartite.get_degree(node) == 1
    for node in stripped.nodes:
        assert stripped.get_degree(node) == bipartite_regular.get_degree(node) - 1
    # Regularity check for original graph(to see if it changes)
    for node in bipartite_regular.nodes:
        assert bipartite_regular.get_degree(node) == bipartite_regular.get_any_degree()
    if not change_origin:
        # Check color does not change in original graph(default color is -1).
        for edge in bipartite_regular.edges:
            assert edge.color != clr
    else:
        origin_edge_clr_cnt = 0
        for edge in bipartite_regular.edges:
            if edge.color == clr:
                origin_edge_clr_cnt += 1
        origin_edge_clr_cnt //= 2
        # print(origin_edge_clr_cnt, match_cnt, len(bipartite_regular.left))
        assert origin_edge_clr_cnt == match_cnt
