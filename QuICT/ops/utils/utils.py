#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/12 下午4:41
# @Author  : Kaiqi Li
# @File    : utils

import numpy as np


def perm_sort(indexes: np.ndarray, blocks: int):
    n = len(indexes)
    iter = n // blocks
    perm_op = []

    # block level swap
    for i in range(blocks):
        for j in range(i + 1, blocks):
            swap = True
            for z in range(iter):
                if indexes[j * iter + z] // iter != i:
                    swap = False
                    break

            if swap:
                perm_op.append(("ALL", i, j))
                indexes[i * iter:i * iter + iter], indexes[j * iter:j * iter + iter] = \
                    indexes[j * iter:j * iter + iter], indexes[i * iter:i * iter + iter]

    for i in range(n):
        block_interval = i // iter
        if indexes[i] // iter != block_interval:
            for j in range(indexes[i] // iter, indexes[i] // iter + iter):
                if indexes[j] // iter == block_interval:
                    break

            indexes[i], indexes[j] = indexes[j], indexes[i]
            perm_op.append(("IDX", i, j))

    return perm_op, indexes
