from typing import *
import numpy as np

from .utility import *


class BlockLDUDecompose:
    @classmethod
    def remapping_select(cls, mat_: np.ndarray) \
            -> List[int]:
        """
        Select some rows out of a square matrix to get a full-ranked sub-matrix.

        Args:
            mat_(np.ndarray): Boolean matrix to be decomposed

        Returns:
            List[int]: Row reordering args.
        """
        mat = mat_.copy()
        n = mat.shape[0]
        l_size = n // 2
        selected = [0]
        unselected = []
        rk = 1
        for i in range(1, n):
            selected.append(i)
            sub_mat: np.ndarray = mat[[selected], :l_size]
            if len(selected) > f2_rank(sub_mat):
                unselected.append(selected.pop())
            else:
                rk += 1
                if rk == l_size:
                    break
        selected.extend(unselected)
        return selected

    @classmethod
    def run(cls, mat_: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Block LDU decompose of a boolean matrix.

        Args:
            mat_(np.ndarray): Boolean matrix to be decomposed

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]: Decomposed L, D, U and row remapping.
        """

        n = mat_.shape[0]
        if n == 1:
            raise Exception("LDU decomposition is designed for matrix whose size > 2.")
        remapping = cls.remapping_select(mat_)
        mat: np.ndarray = mat_[remapping].copy()
        l_size = n // 2
        r_size = n - l_size
        a = mat[:l_size, :l_size]
        b = mat[:l_size, l_size:]
        c = mat[l_size:, :l_size]
        d_ = mat[l_size:, l_size:]
        a_inv = f2_inverse(a)
        c_a_inv = f2_prod(c, a_inv)
        c_a_inv_b = f2_prod(c_a_inv, b)
        a_inv_b = f2_prod(a_inv, b)
        l_ = np.block([
            [np.eye(l_size, dtype=bool), np.zeros(shape=(l_size, r_size), dtype=bool)],
            [c_a_inv, np.eye(r_size, dtype=bool)]
        ])
        d_ = np.block([
            [a, np.zeros(shape=(l_size, r_size), dtype=bool)],
            [np.zeros(shape=(r_size, l_size), dtype=bool), d_ - c_a_inv_b]
        ])
        u_ = np.block([
            [np.eye(l_size, dtype=bool), a_inv_b],
            [np.zeros(shape=(r_size, l_size), dtype=bool), np.eye(r_size, dtype=bool)]
        ])
        return l_, d_, u_, remapping
