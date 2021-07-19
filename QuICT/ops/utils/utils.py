#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 下午4:41
# @Author  : Kaiqi Li
# @File    : utils

import numpy as np
from numba import njit


@njit(nogil=True)
def mapping_augment(mapping: np.ndarray) -> np.ndarray:
    n = len(mapping)
    p2n = 1 << n
    res = np.zeros(shape=p2n, dtype=np.int64)
    for i in range(p2n):
        for k in range(n):
            res[i] |= ((i >> (n - 1 - mapping[k])) & 1) << (n - 1 - k)
    return res
