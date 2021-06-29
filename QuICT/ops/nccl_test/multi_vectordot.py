
import cupy as cp
import numpy as np
from cupy import cuda
from cupy.cuda import nccl
import multiprocessing

from QuICT.ops.linalg.gpu_calculator import *
from QuICT.ops.linalg.proxy import Proxy


def client(uid, dev_id):
    proxy = Proxy(ndevs=2, uid=uid, rank=dev_id)

    based_data = np.arange(10)*2
    based_data = based_data.astype(np.complex64)

    gpu_bd = cp.array(based_data)

    print(gpu_bd)

    recvbuf = cp.zeros(5, dtype=cp.complex64)
    # proxy.broadcast(recvbuf, 0, cuda.Stream.null.ptr)

    proxy.send(gpu_bd[5:], 0, cuda.Stream.null.ptr)

    print("finish send")

    proxy.recv(recvbuf, 0, cuda.Stream.null.ptr)

    gpu_bd[5:] = recvbuf

    print(gpu_bd)


def server(uid):
    proxy = Proxy(ndevs=2, uid=uid, rank=0)

    based_data = np.arange(10)*5
    based_data = based_data.astype(np.complex64)

    gpu_bd = cp.array(based_data)

    print(gpu_bd)

    recvbuf = cp.zeros(5, dtype=cp.complex64)

    # broadcast A matrix
    # proxy.broadcast(A, 0, cuda.Stream.null.ptr)

    proxy.send(gpu_bd[:5], 1, cuda.Stream.null.ptr)

    print("finish send")

    proxy.recv(recvbuf, 1, cuda.Stream.null.ptr)

    gpu_bd[:5] = recvbuf

    print(gpu_bd)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    uid = nccl.get_unique_id()

    p1 = multiprocessing.Process(target=server, args = (uid,))
    p2 = multiprocessing.Process(target=client, args = (uid, 1,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
