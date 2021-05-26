from QuICT.utility.graph_structure import GraphStructureBuilder


def test_graph():
    print()
    g = GraphStructureBuilder()

    g.add_edge(1, 2, 102)
    g.add_edge(2, 3, 203)
    g.add_edge(3, 4, 304)

    g.print_info()

    for e in g.edges_from(1):
        e.print_info()

