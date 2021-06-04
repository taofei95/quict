from typing import *
from time import time

import numpy as np
import numba as nb
from numba import cuda as nb_cuda

from QuICT.core import *

thread_num_per_block = 128  # no structure


@nb_cuda.jit()
def _gate_on_state_vector_kernel(
        state_vector,
        affect_num,
        mat_offset,
        arg_offset,
        compact_mat,
        compact_arg,
        compact_arg_sorted,
):
    index: int
    mat_bit: int
    index = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x

    mat_n_rows = 1 << affect_num

    gate_id_in_blk: int = nb_cuda.threadIdx.x
    mat_bit = nb_cuda.threadIdx.y

    vec_cache_blk_shared = nb_cuda.shared.array(shape=64, dtype=nb.complex64)  # 512B

    cache_offset = gate_id_in_blk * mat_n_rows

    for i in range(affect_num):
        pos = compact_arg_sorted[arg_offset + i]
        tail = index & ((1 << pos) - 1)
        index = index >> pos << (pos + 1) | tail
    for i in range(affect_num):
        pos = compact_arg[arg_offset + i]
        val = mat_bit >> (affect_num - 1 - i) & 1
        index |= val << pos

    vec_cache_blk_shared[cache_offset + mat_bit] = state_vector[index]

    nb_cuda.syncthreads()

    tmp = nb.complex64(0.0 + 0.0j)

    compact_mat_offset = mat_offset + mat_bit * mat_n_rows
    for i in range(mat_n_rows):
        tmp += compact_mat[compact_mat_offset + i] * vec_cache_blk_shared[cache_offset + i]

    state_vector[index] = tmp


def state_vector_simulation(
        initial_state: np.ndarray,
        gates: List[BasicGate],
):
    with nb_cuda.gpus[0]:

        start_time = time()

        mat_offset = 0
        arg_offset = 0
        gate_cnt = len(gates)
        mat_strides = np.empty(shape=gate_cnt, dtype=np.int64)
        arg_strides = np.empty(shape=gate_cnt, dtype=np.int64)

        for i, gate in enumerate(gates):
            mat_strides[i] = mat_offset
            arg_strides[i] = arg_offset
            arg_offset += len(gate.affectArgs)
            mat_offset += gate.compute_matrix.shape[0] ** 2

        compact_mat = np.empty(shape=mat_offset, dtype=np.complex64)
        compact_arg = np.empty(shape=arg_offset, dtype=np.int64)
        compact_arg_sorted = np.empty(shape=arg_offset, dtype=np.int64)

        for i, gate in enumerate(gates):
            offset = mat_strides[i]
            mat_sz = gate.compute_matrix.shape[0] ** 2
            compact_mat[offset:offset + mat_sz] = gate.compute_matrix.flatten()[:]
            offset = arg_strides[i]
            affect_num = len(gate.affectArgs)
            tmp_arr = np.array(gate.affectArgs)
            compact_arg[offset:offset + affect_num] = tmp_arr[:]
            compact_arg_sorted[offset:offset + affect_num] = np.sort(tmp_arr)[:]

        end_time = time()
        data_gather_time = end_time - start_time

        start_time = time()

        d_state_vector = nb_cuda.to_device(initial_state)
        d_compact_mat = nb_cuda.to_device(compact_mat)
        d_compact_arg = nb_cuda.to_device(compact_arg)
        d_compact_arg_sorted = nb_cuda.to_device(compact_arg_sorted)

        end_time = time()
        data_transfer_time = end_time - start_time

        start_time = time()

        for i, gate in enumerate(gates):
            affect_num = len(gate.affectArgs)
            mat_n_rows = gate.compute_matrix.shape[0]
            gate_in_blk = thread_num_per_block // mat_n_rows
            thread_per_block = (gate_in_blk, mat_n_rows)
            blk_cnt = initial_state.shape[0] // thread_num_per_block
            _gate_on_state_vector_kernel[blk_cnt, thread_per_block](
                d_state_vector,
                affect_num,
                mat_strides[i],
                arg_strides[i],
                d_compact_mat,
                d_compact_arg,
                d_compact_arg_sorted,
            )

        end_time = time()
        cuda_run_time = end_time - start_time

        print()
        print(f"Data gather time: {data_gather_time:0.4f} s")
        print(f"Data transfer time: {data_transfer_time:0.4f} s")
        print(f"CUDA execution time: {cuda_run_time:0.4f} s")

        # return d_state_vector.copy_to_host()
