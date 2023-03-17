import numpy as np


class TridiagonalToeplitz:
    """ the matrices like
        [a  b' 0  0  ...  0 ]
        |b  a  b' 0  ...  0 |
        |0  b  a  b' 0..  0 |
        |0  .  b  a  b'.  0 |
        |.  0  0  ..b  a  b'|
        [0  0  0  ...  b  a ]
        which a called main diagonal and b called off diagonal.
        These kinds of hermitian matrices are usually good to be simulated
    """
    def __init__(self, size, main_diag, off_diag) -> None:
        """
        Args:
            size(int): size of the matrix
            main_diag(float/complex): the elements in main diagonal
            off_diag(float/complex): the elements in off diagonal
        """
        self.size = size
        self.main_diag = main_diag
        self.off_diag = off_diag

    def matrix(self):
        """
        Return:
            ndarray: the tridiagnal toeplitz matrix
        """
        n = self.size
        m = self.main_diag * np.identity(n, dtype=np.complex128)
        for idx in range(1, n):
            m[idx - 1, idx] = self.off_diag
            m[idx, idx - 1] = self.off_diag
        return m


TM = TridiagonalToeplitz
