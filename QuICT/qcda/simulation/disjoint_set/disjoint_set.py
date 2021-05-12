import numpy as np


class DisjointSet:
    def __init__(self, total_cnt: int):
        self._father = [i for i in range(total_cnt)]
        self._rank = np.ones(shape=total_cnt, dtype=np.int64)

    def find(self, x: int) -> int:
        while self._father[x] != x:
            self._father[x] = self._father[self._father[x]]
            x = self._father[x]
        return x

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
