
import cupy as cp
import numpy as np
from cupy import cuda
from cupy.cuda import nccl
import multiprocessing

from QuICT.ops.linalg.gpu_calculator import *
from QuICT.ops.linalg.proxy import Proxy


def client(uid, dev_id):
    proxy = Proxy(ndevs=3, uid=uid, rank=dev_id)

    based_data = np.arange(5).astype(np.complex64)

    gpu_bd = cp.array(based_data)

    recvbuf = cp.zeros(5, dtype=cp.complex64)
    # proxy.broadcast(recvbuf, 0, cuda.Stream.null.ptr)

    proxy.recv(recvbuf, 0)

    print(f"finish reduce. {recvbuf} \n")

    # proxy.recv(recvbuf, 0, cuda.Stream.null.ptr)

    # gpu_bd[5:] = recvbuf

    # print(gpu_bd)


def server(uid):
    proxy = Proxy(ndevs=3, uid=uid, rank=0)

    based_data = np.arange(10).astype(np.complex64)

    gpu_bd = cp.array(based_data)

    recvbuf = cp.zeros(10, dtype=cp.complex64)

    # broadcast A matrix
    # proxy.broadcast(gpu_bd, 0, cuda.Stream.null.ptr)

    # Reduce
    # proxy.allreduce(gpu_bd, recvbuf, 0)
    # proxy.reducescatter(gpu_bd, recvbuf, 2)

    # Gather
    # proxy.allgather(gpu_bd, recvbuf)
    # proxy.gather(0, None, recvbuf)

    # Scatter
    proxy.scatter(gpu_bd, [1])

    # proxy.send(gpu_bd[:5], 1, cuda.Stream.null.ptr)

    print(f"finish receive {gpu_bd} \n")

    # proxy.recv(recvbuf, 1, cuda.Stream.null.ptr)

    # gpu_bd[:5] = recvbuf

    # print(gpu_bd)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    uid = nccl.get_unique_id()

    p1 = multiprocessing.Process(target=server, args = (uid,))
    p2 = multiprocessing.Process(target=client, args = (uid, 1,))
    p3 = multiprocessing.Process(target=client, args = (uid, 2,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
