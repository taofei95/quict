from typing import *
import numpy as np


class BlockLDUDecompose:
    @classmethod
    def f2_half_gaussian_elimination(cls, mat_: np.ndarray) -> np.ndarray:
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

    @classmethod
    def f2_rank(cls, mat_: np.ndarray) -> int:
        mat = cls.f2_half_gaussian_elimination(mat_)
        rk = 0
        n = min(mat.shape[0], mat.shape[1])
        for i in range(n):
            if mat[i, i]:
                rk += 1
        return rk

    @classmethod
    def remapping_select(cls, mat_: np.ndarray) \
            -> List[int]:
        """
        Select some rows to get a full-ranked sub-matrix.

        Args:
            mat_(np.ndarray): Boolean matrix to be decomposed

        Returns:
            List[int]: Selected rows.
        """
        mat = mat_.copy()
        n = mat.shape[0]
        l_size = n // 2
        selected = [0]
        rk = 1
        for i in range(1, n):
            selected.append(i)
            sub_mat: np.ndarray = mat[[selected], :l_size]
            if l_size == cls.f2_rank(sub_mat):
                selected.pop()
            else:
                rk += 1
                if rk == l_size:
                    break
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

        mat = mat_.copy()
        n = mat.shape[0]
        if n == 1:
            raise Exception("LDU decomposition is designed for matrix whose size > 2.")
