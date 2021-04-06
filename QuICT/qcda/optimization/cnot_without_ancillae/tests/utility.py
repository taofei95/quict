import numpy as np
import random

from ..graph import *


def f2_random_invertible_matrix_gen(n) -> np.ndarray:
    mat = np.eye(n, dtype=bool)
    rnd = 20 * n
    rg_lst = list(range(n))
    for _ in range(rnd):
        x = random.sample(rg_lst, 2)
        i = x[0]
        j = x[1]
        mat[i, :] ^= mat[j, :]
    return mat


def ensure_bipartite_max_degree_even(bipartite: Bipartite) -> None:
    if bipartite.get_max_degree() % 2 == 1:
        mx_deg_node = bipartite.get_max_degree_node()
        another = bipartite.right[0] if mx_deg_node in bipartite.left else bipartite.left[0]
        bipartite.add_edge(mx_deg_node, another)
        bipartite.add_edge(another, mx_deg_node)


def get_bipartite(size: int = 150, ensure_even: bool = True) -> Bipartite:
    split_point = size + 1
    delta = int(size / 10)
    split_point += random.randint(-delta, delta)
    left = list(range(1, size + 1))
    right = list(range(size + 1, size * 2))
    bipartite = Bipartite(left, right)
    f = random.random() * 0.9 + 0.1
    for _ in range(int(size * size * f)):
        a = random.choice(left)
        b = random.choice(right)
        bipartite.add_edge(a, b)
        bipartite.add_edge(b, a)

    if ensure_even:
        ensure_bipartite_max_degree_even(bipartite)

    return bipartite


def _b_bipartite_test(_b: Bipartite):
    for node in _b.left:
        eid = _b.head[node]
        while eid != -1:
            edge = _b.edges[eid]
            # print(f"node={node}, start={edge.start}, end={edge.end}, eid={eid}")
            assert edge.start == node
            assert edge.end in _b.right

            eid = edge.next
    for node in _b.right:
        eid = _b.head[node]
        while eid != -1:
            edge = _b.edges[eid]
            assert edge.start == node
            assert edge.end in _b.left
            # print(f"node={node}, start={edge.start}, end={edge.end}")
            eid = edge.next
