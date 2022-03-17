import numpy as np


def is_kraus_ops(kraus: list, based=None) -> bool:
    row, col = kraus[0].shape
    n = int(np.log2(row)) if based is None else based
    if not row == col or not row == 2 ** n:
        return False

    for kmat in kraus:
        if not kmat.shape == (row, col):
            return False

    kk = sum(np.transpose(k).conjugate().dot(k) for k in kraus)
    if not np.allclose(kk, np.identity(row, dtype=kraus.dtype)):
        return False

    return True
