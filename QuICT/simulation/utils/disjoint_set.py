import numpy as np


class DisjointSet:
    def __init__(self, total_cnt: int):
        self._father = np.array([i for i in range(total_cnt)])
        self._rank = np.ones(shape=total_cnt, dtype=np.int64)

    def find(self, x: int) -> int:
        if x == self._father[x]:
            return x
        self._father[x] = self.find(self._father[x])
        return self._father[x]

    def union(self, x: int, y: int):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return x

        if self._rank[x] < self._rank[y]:
            x, y = y, x

        self._father[y] = x
        if self._rank[x] == self._rank[y]:
            self._rank[x] += 1
        return x
