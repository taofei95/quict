import numpy as np
from scipy import stats, sparse


class RandomSparse:
    def __init__(self, size, density) -> None:
        self.size = size
        self.density = density

    def matrix(self):
        size = self.size
        rvs = stats.norm().rvs
        while(1):
            X = sparse.random(
                size, size, density=self.density, data_rvs=rvs,
                dtype=np.complex128)
            A = X.todense()
            v = np.linalg.eigvals(A)
            A = np.round(A, 4)
            if np.linalg.det(A) != 0 and np.log2(max(abs(v)) / min(abs(v))) < 6:
                return np.array(A)


RS = RandomSparse
