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
        x = self.nodes + other.nodes
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


def f2_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_ = np.array(a, dtype=int)
    b_ = np.array(b, dtype=int)
    c_ = a_ @ b_
    c_ %= 2
    c = np.array(c_, dtype=bool)
    return c


def f2_half_gaussian_elimination(mat_: np.ndarray) -> np.ndarray:
    mat: np.ndarray = mat_.copy()
    n = min(mat.shape[0], mat.shape[1])
    for i in range(n):
        if not mat[i, i]:
            for j in range(i + 1, mat.shape[0]):
                if mat[j, i]:
                    mat[[i, j], :] = mat[[j, i], :]
                    break
        if mat[i, i]:
            for j in range(i + 1, mat.shape[0]):
                if mat[j, i]:
                    mat[j, :] ^= mat[i, :]
    return mat


def f2_rank(mat_: np.ndarray) -> int:
    mat = f2_half_gaussian_elimination(mat_)
    rk = 0
    n = min(mat.shape[0], mat.shape[1])
    for i in range(n):
        if mat[i, i]:
            rk += 1
    return rk


def f2_inverse(mat_: np.ndarray) -> np.ndarray:
    n = mat_.shape[0]
    aug = np.empty(shape=(n, 2 * n), dtype=bool)
    aug[:, :n] = mat_
    aug[:, n:] = np.eye(n, dtype=bool)

    # gaussian elimination
    for i in range(n):
        if not aug[i, i]:
            for k in range(i + 1, n):
                if aug[k, i]:
                    aug[[k, i], :] = aug[[i, k], :]
                    break
        for k in range(n):
            if i != k and aug[k, i]:
                aug[k, :] ^= aug[i, :]
    ret = aug[:, n:].copy()
    return ret
