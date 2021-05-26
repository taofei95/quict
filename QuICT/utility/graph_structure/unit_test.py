from QuICT.utility.graph_structure import directed_graph_builder


def test_graph():
    print()
    g = directed_graph_builder()

    g.add_edge(1, 2, 102)
    g.add_edge(2, 3, 203)
    g.add_edge(3, 4, 304)

    g.print_info()

    for e in g.edges_from(1):
        e.print_info()


def test_special_label():
    g = directed_graph_builder(tuple)
    g.add_edge((1, 2), (2, 1), 0)
    g.add_edge((1, 3), (3, 1), 1)

    g.print_info()
