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
    """ The proxy class of NCCL Communicators, which is used to transfer data between gpus.
    
    Args:
        ndevs(int): number of total GPU devices.
        uid(tuple): The unique ID, generate by cupy.cuda.nccl.get_unique_id().
        rank(int): The rank of GPU, between 0 and ndevs-1; represent the gpu device use in current process.
     """
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
        """ Initialize NCCL communicators for multiple devices in a single process. """
        return nccl.NcclCommunicator.initAll(devs)

    @property
    def ndevs(self):
        """ return the number of GPU devices. """
        return self._ndevs

    @property
    def uid(self):
        """ return the unique id of current communicator. """
        return self._uid

    @property
    def rank(self):
        """ return the rank of current communicator. """
        return self._rank

    def send(
        self,
        sendbuf: cp.ndarray,
        destination: int,
        stream: cuda.stream.Stream = cuda.Stream.null.ptr
    ):
        """ Send data to destination.
        
        Args:
            sendbuf(cupy.ndarray): the sending data.
            destination(int): the rank of destination communicator.
            stream(Stream): the cupy stream.
        """
        assert(type(sendbuf) == cp.ndarray)
        pointer = sendbuf.data.ptr
        count = sendbuf.size
        nccl_datatype = type_mapping[str(sendbuf.dtype)]

        if sendbuf.dtype == cp.complex128:
            count *= 2

        self.comm.send(pointer, count, nccl_datatype, destination, stream)

    def recv(
        self,
        recvbuf: cp.ndarray,
        source: int,
        stream: cuda.stream.Stream = cuda.Stream.null.ptr
    ):
        """ recv data from source and put it into recvbuf.

        Args:
            recvbuf(cupy.ndarray): the empty array waitting for comming data.
            source(int): the rank of the source communicator.
            stream(Stream): the cupy stream.
        """
        count = recvbuf.size
        nccl_datatype = type_mapping[str(recvbuf.dtype)]

        if recvbuf.dtype == cp.complex128:
            count *= 2

        self.comm.recv(recvbuf.data.ptr, count, nccl_datatype, source, stream)

        return recvbuf

    def broadcast(
        self,
        sendbuf,
        root: int = 0,
        stream: cuda.stream.Stream = cuda.Stream.null.ptr
    ):
        """ Send data to all communicators except root. The sending data is in sendbuf of root communicator.

        Args:
            sendbuf(cupy.ndarray): the sending data in root, and received array for other communicators.
            root(int): the rank of root communicator.
            stream(Stream): the cupy stream.
        """
        pointer = sendbuf.data.ptr
        count = sendbuf.size
        nccl_datatype = type_mapping[str(sendbuf.dtype)]

        if sendbuf.dtype == cp.complex128:
            count *= 2

        self.comm.bcast(sendbuf.data.ptr, count, nccl_datatype, root, stream)
