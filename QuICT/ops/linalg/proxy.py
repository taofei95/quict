import cupy as cp
import numpy as np
from cupy import cuda
from cupy.cuda import nccl

from typing import Union


type_mapping = {
    "int8": nccl.NCCL_INT8,
    "uint8": nccl.NCCL_UINT8,
    "int32": nccl.NCCL_INT32,
    "uint32": nccl.NCCL_UINT32,
    "int64": nccl.NCCL_INT64,
    "uint64": nccl.NCCL_UINT64,
    "float16": nccl.NCCL_FLOAT16,
    "float32": nccl.NCCL_FLOAT32,
    "float64": nccl.NCCL_FLOAT64,
    "complex64": nccl.NCCL_FLOAT64,
    "complex128": nccl.NCCL_FLOAT64
}

class Proxy:
    """ """
    def __init__(self, ndevs: int, uid: tuple, rank: int):
        assert((rank < ndevs) and (ndevs == cuda.runtime.getDeviceCount()))
        if rank != cuda.runtime.getDevice():
            target_device = cuda.Device(rank)
            target_device.use()

        self._ndevs = ndevs
        self._uid = uid
        self._rank = rank
        self.comm = nccl.NcclCommunicator(self._ndevs, self._uid, self._rank)

    def __exit__(self):
        self.comm.destroy()

    @staticmethod
    def allInit(devs: Union[int, list]):
        return nccl.NcclCommunicator.initAll(devs)

    @property
    def ndevs(self):
        return self._ndevs

    @property
    def uid(self):
        return self._uid

    @property
    def rank(self):
        return self._rank

    def send(
        self,
        sendbuf,
        destination: int,
        stream: cuda.stream.Stream = cuda.Stream.null.ptr
    ):
        assert(type(sendbuf) == cp.ndarray)
        pointer = sendbuf.data.ptr
        count = sendbuf.size
        nccl_datatype = type_mapping[str(sendbuf.dtype)]

        if sendbuf.dtype == cp.complex128:
            count *= 2

        self.comm.send(pointer, count, nccl_datatype, destination, stream)

    def recv(
        self,
        recv_shape: tuple,
        recv_type: type,
        source: int,
        stream: cuda.stream.Stream = cuda.Stream.null.ptr
    ):
        recvbuf = cp.empty(recv_shape, dtype=recv_type)
        count = recvbuf.size
        nccl_datatype = type_mapping[str(recv_type)]

        if recv_type == cp.complex128:
            count *= 2

        self.comm.recv(recvbuf.data.ptr, count, nccl_datatype, source, stream)

        return recvbuf

    def broadcast(
        self,
        sendbuf,
        root: int = 0,
        stream: cuda.stream.Stream = cuda.Stream.null.ptr
    ):
        """ Sendbuf in root device, otherwise, it is receive buffer. """
        pointer = sendbuf.data.ptr
        count = sendbuf.size
        nccl_datatype = type_mapping[str(sendbuf.dtype)]

        if sendbuf.dtype == cp.complex128:
            count *= 2

        self.comm.bcast(sendbuf.data.ptr, count, nccl_datatype, root, stream)
