import numpy as np
from typing import *

from .utility import *
from .graph import *
from .edge_coloring import EdgeColoring


def test_bipartite_construction(
        bipartite: Bipartite,
        mat: np.ndarray,
        is_lower_triangular: bool
) -> bool:
    def has_edge(_b: Bipartite, s: int, t: int) -> bool:
        eid = _b.head[s]
        while eid != -1:
            edge = _b.edges[eid]
            if edge.end == t:
                return True
            eid = edge.next
        return False

    n = mat.shape[0]
    s_size = n // 2
    t_size = n - s_size

    if is_lower_triangular:
        for j in range(s_size):
            for i in range(t_size):
                if mat[i + s_size, j] and (not has_edge(bipartite, j, i + s_size)):
                    return False
    else:
        for j in range(t_size):
            for i in range(s_size):
                if mat[i, j + s_size] and (not has_edge(bipartite, j, i + t_size)):
                    return False

    return True


class CnotWithoutAncillae:

    @classmethod
    def matrix_run(cls, mat: np.ndarray) \
            -> List[List[Tuple[int, int]]]:
        """
        Get parallel row elimination of a boolean invertible matrix.

        Args:
            mat (np.ndarray): A boolean invertible matrix

        Returns:
            List[List[Tuple[int, int]]]: Parallel row elimination. In each depth level there are
            non-overlapping row eliminations.
        """
        if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
            raise Exception("Must use a matrix!")
        if mat.dtype is not bool:
            raise Exception("Must use a boolean matrix!")

        pass

    @classmethod
    def triangular_matrix_run(cls, mat: np.ndarray, is_lower_triangular: bool) \
            -> List[List[Tuple[int, int]]]:
        """
        Get parallel row elimination of a triangular matrix by bipartite edge coloring.


        Args:
            mat (np.ndarray): A triangular boolean matrix with block identity diagonal entries.
            is_lower_triangular (bool): Is the input triangular matrix has all its non-zero
                entries in lower part. If this parameter is false, all non-zero entries should
                be located in upper part.

        Returns:
            List[List[Tuple[int, int]]]: Parallel row elimination. In each depth level there are
            non-overlapping row eliminations.
        """
        n = mat.shape[0]
        s_size = n // 2
        t_size = n - s_size

        # Always use the identity sub matrix as the `left` in bipartite
        if is_lower_triangular:
            left = [i for i in range(s_size)]
            right = [i + s_size for i in range(t_size)]
        else:
            left = [i for i in range(t_size)]
            right = [i + t_size for i in range(s_size)]

        # Construct bipartite
        bipartite = Bipartite(left, right)
        # Add edges
        if is_lower_triangular:
            for j in range(s_size):
                for i in range(s_size, n):
                    if mat[i, j]:
                        bipartite.add_edge(j, i)
                        bipartite.add_edge(i, j)
        else:
            for j in range(s_size, n):
                for i in range(s_size):
                    if mat[i, j]:
                        bipartite.add_edge(j - s_size, i + t_size)
                        bipartite.add_edge(i + t_size, j - s_size)

        assert test_bipartite_construction(bipartite, mat, is_lower_triangular)

        colored_bipartite = EdgeColoring.get_edge_coloring(bipartite)
        max_deg = colored_bipartite.get_max_degree()
        parallel_elimination: List[List[Tuple[int, int]]] = [[] for _ in range(max_deg)]

        # Iterate over left vertices to check edges with the same color
        for node in colored_bipartite.left:
            eid = colored_bipartite.head[node]
            while eid != -1:
                edge = colored_bipartite.edges[eid]
                color_index = edge.color
                if is_lower_triangular:
                    c = edge.start
                    t = edge.end
                else:
                    c = edge.start + s_size
                    t = edge.end - t_size
                row_elimination = (c, t)
                parallel_elimination[color_index].append(row_elimination)
                eid = edge.next

        return parallel_elimination

    @classmethod
    def block_diagonal_matrix_run(cls, mat: np.ndarray) \
            -> List[List[Tuple[int, int]]]:
        """
        Get parallel row elimination of a block diagonal matrix by bipartite edge coloring.


        Args:
            mat (np.ndarray): A block diagonal boolean matrix.

        Returns:
            List[List[Tuple[int, int]]]: Parallel row elimination. In each depth level there are
            non-overlapping row eliminations.
        """
        pass
