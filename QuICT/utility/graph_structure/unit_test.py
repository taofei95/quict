from QuICT.utility.graph_structure import DirectedGraph, Vertex


def test_graph():
    print()
    g = DirectedGraph()

    v1 = Vertex(1, "v_1")
    v2 = Vertex(3, "v_3")
    v3 = Vertex(5, "v_5")
    v4 = Vertex(7, "v_7")

    g.add_edge(v1, v2, 102) \
        .add_edge(v1, v3, 103) \
        .add_edge(v1, v4, 104) \
        .add_edge(v2, v3, 203) \
        .add_edge(v3, v4, 304)

    g.print_info()

    print(g.out_deg_of(v1))
    print(g.in_deg_of(v4))

    for e in g.edges_from(v1):
        e.print_info()

    for e in g.edges_to(v4):
        e.print_info()
