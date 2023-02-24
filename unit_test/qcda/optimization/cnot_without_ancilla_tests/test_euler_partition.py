import random

from .utility import get_bipartite, _b_bipartite_test
from QuICT.qcda.optimization.cnot_without_ancilla import *


def test_euler_partition():
    bipartite = get_bipartite(random.randint(100, 200))
    _, _, bipartite_regular = EdgeColoring.get_regular(bipartite)
    b1, b2 = EdgeColoring.get_euler_partition(bipartite_regular)
    # check bipartite
    _b_bipartite_test(b1)
    _b_bipartite_test(b2)

    # check regularity
    d = b1.get_any_degree()
    for node in b1.nodes:
        assert d == b1.get_degree(node)
    for node in b2.nodes:
        assert d == b2.get_degree(node)
    # check if half split
    assert len(b1.edges) == len(b2.edges)
    assert len(b1.edges) * 2 == len(bipartite_regular.edges)
    assert b1.get_any_degree() * 2 == bipartite_regular.get_max_degree()
