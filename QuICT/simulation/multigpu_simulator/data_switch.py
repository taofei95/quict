import numpy as np
import cupy as cp
import math
import os


LIMIT_BUFFER_SIZE = int(os.getenv("QuICT_BUFFER_SIZE", 17))


class DataSwitcher:
    def __init__(self, proxy, qubits: int, precision=np.complex64):
        self._proxy = proxy
        self._qubits = qubits
        self._based_idx = np.arange(1 << qubits, dtype=np.int64)
        self._id = proxy.rank
        if precision == np.complex64:
            self._max_data_size_per_time = 1 << LIMIT_BUFFER_SIZE
        else:
            self._max_data_size_per_time = 1 << (LIMIT_BUFFER_SIZE - 1)

    def _switch(self, vector, destination):
        recv_buf = cp.zeros(vector.size, dtype=vector.dtype)
        sliced_data = min(vector.size, self._max_data_size_per_time)

        for i in range(math.ceil(vector.size / sliced_data)):
            self._proxy.send(vector[i * sliced_data:(i + 1) * sliced_data], destination)

            self._proxy.recv(recv_buf[i * sliced_data:(i + 1) * sliced_data], destination)

        return recv_buf

    def all_switch(self, vector, destination):
        vector[:] = self._switch(vector, destination)

    def half_switch(self, vector, destination):
        _0_1 = self._id < destination
        sending_size = vector.size // 2

        if not _0_1:
            sending_data = vector[:sending_size]
            recv_buf = self._switch(sending_data, destination)
            vector[:sending_size] = recv_buf
        else:
            sending_data = vector[sending_size:]
            recv_buf = self._switch(sending_data, destination)
            vector[sending_size:] = recv_buf

    def ctargs_switch(self, vector, destination: int, condition: dict):
        """
            condition: dict{index, 0/1}
        """
        target_idx = self._based_idx
        for idx, _0_1 in condition.items():
            if isinstance(target_idx, tuple):
                target_idx = target_idx[0]

            if _0_1:
                target_idx = target_idx[np.where(target_idx & (1 << idx))]
            else:
                target_idx = target_idx[np.where((target_idx & (1 << idx)) == 0)]

        sending_data = vector[target_idx]
        recv_buf = self._switch(sending_data, destination)
        vector[target_idx] = recv_buf

    def add_prob(self, prob_result):
        recv_buf = cp.zeros_like(prob_result)

        self._proxy.allreduce(
            sendbuf=prob_result,
            recvbuf=recv_buf,
            op=0
        )

        return recv_buf
