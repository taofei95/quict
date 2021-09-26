#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/7/2 下午2:41
# @Author  : Kaiqi Li
# @File    : broadcast

import cupy as cp
import numpy as np
from cupy import cuda
from cupy.cuda import nccl
import multiprocessing

from QuICT.ops.utils import Proxy


def client(uid, dev_id):
    proxy = Proxy(ndevs=3, uid=uid, rank=dev_id)
    recvbuf = cp.zeros(15, dtype=cp.complex64)

    # Broadcast
    proxy.broadcast(recvbuf, 0, cuda.Stream.null.ptr)

    print(f"Receiving through broadcast: {recvbuf}")


def server(uid):
    proxy = Proxy(ndevs=3, uid=uid, rank=0)
    based_data = np.arange(15).astype(np.complex64)
    gpu_bd = cp.array(based_data)

    # broadcast A matrix
    proxy.broadcast(gpu_bd, 0, cuda.Stream.null.ptr)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    uid = nccl.get_unique_id()

    p1 = multiprocessing.Process(target=server, args=(uid,))
    p2 = multiprocessing.Process(target=client, args=(uid, 1,))
    p3 = multiprocessing.Process(target=client, args=(uid, 2,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
