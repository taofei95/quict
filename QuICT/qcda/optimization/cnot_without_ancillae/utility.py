#!/usr/bin/env python
# -*- coding:utf8 -*-

from typing import *
import numpy as np


class Merged:
    """merged points in heap

    Attributes
    ----------
    deg : int
        degree of x
    nodes : List[int]
        node indices in a bipartite
    """

    def __init__(self, deg: int, nodes: List[int]):
        """get a merged point

        Parameters
        ----------
        deg : int
            degree of x
        nodes : List[int]
            node indices in a bipartite
        """
        self.deg = deg
        self.nodes = nodes

    def __add__(self, other):
        """combine two merged points to a larger merged

        Parameters
        ----------
        other : Merged

        Returns
        -------
        Merged
        """
        d = self.deg + other.deg
        x = []
        x.extend(self.nodes)
        x.extend(other.nodes)
        return Merged(d, x)

    def __lt__(self, other):
        """self is less than other

        Parameters
        ----------
        other : Merged

        Returns
        -------
        Boolean
        """
        return self.deg < other.deg


def is_not_identity(mat) -> bool:
    n = mat.shape[0]
    return not np.all(mat == np.eye(n, dtype=bool))
