from typing import *
from time import time

import numpy as np
import numba as nb
from numba import cuda as nb_cuda

from QuICT.core import *

thread_num_per_block = 128  # no structure


# @nb_cuda.jit()
# def _gate_on_state_vector_kernel_1bit(
#         state_vector,
#         mat_offset,
#         arg_offset,
#         compact_mat,
#         compact_arg,
#         compact_arg_sorted,
# ):
#     index: int
#     mat_bit: int
#     index = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x
#
#     mat_n_rows = 2
#
#     gate_id_in_blk: int = nb_cuda.threadIdx.x
#     mat_bit = nb_cuda.threadIdx.y
#
#     vec_cache_blk_shared = nb_cuda.shared.array(shape=64, dtype=nb.complex64)  # 512B
#
#     cache_offset = gate_id_in_blk * mat_n_rows
#
#     pos = compact_arg_sorted[arg_offset]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit & 1
#     index |= val << pos
#
#     vec_cache_blk_shared[cache_offset + mat_bit] = state_vector[index]
#
#     nb_cuda.syncthreads()
#
#     tmp = nb.complex64(0.0 + 0.0j)
#
#     compact_mat_offset = mat_offset + mat_bit * mat_n_rows
#     tmp += compact_mat[compact_mat_offset] * vec_cache_blk_shared[cache_offset]
#     tmp += compact_mat[compact_mat_offset + 1] * vec_cache_blk_shared[cache_offset + 1]
#
#     state_vector[index] = tmp
#
#
# @nb_cuda.jit()
# def _gate_on_state_vector_kernel_2bit(
#         state_vector,
#         mat_offset,
#         arg_offset,
#         compact_mat,
#         compact_arg,
#         compact_arg_sorted,
# ):
#     index: int
#     mat_bit: int
#     index = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x
#
#     mat_n_rows = 4
#
#     gate_id_in_blk: int = nb_cuda.threadIdx.x
#     mat_bit = nb_cuda.threadIdx.y
#
#     vec_cache_blk_shared = nb_cuda.shared.array(shape=64, dtype=nb.complex64)  # 512B
#
#     cache_offset = gate_id_in_blk * mat_n_rows
#
#     pos = compact_arg_sorted[arg_offset + 0]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg_sorted[arg_offset + 1]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit >> 1 & 1
#     index |= val << pos
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit & 1
#     index |= val << pos
#
#     vec_cache_blk_shared[cache_offset + mat_bit] = state_vector[index]
#
#     nb_cuda.syncthreads()
#
#     tmp = nb.complex64(0.0 + 0.0j)
#
#     compact_mat_offset = mat_offset + mat_bit * mat_n_rows
#     tmp += compact_mat[compact_mat_offset] * vec_cache_blk_shared[cache_offset]
#     tmp += compact_mat[compact_mat_offset + 1] * vec_cache_blk_shared[cache_offset + 1]
#     tmp += compact_mat[compact_mat_offset + 2] * vec_cache_blk_shared[cache_offset + 2]
#     tmp += compact_mat[compact_mat_offset + 3] * vec_cache_blk_shared[cache_offset + 3]
#
#     state_vector[index] = tmp
#
#
# @nb_cuda.jit()
# def _gate_on_state_vector_kernel_2bit(
#         state_vector,
#         mat_offset,
#         arg_offset,
#         compact_mat,
#         compact_arg,
#         compact_arg_sorted,
# ):
#     index: int
#     mat_bit: int
#     index = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x
#
#     mat_n_rows = 4
#
#     gate_id_in_blk: int = nb_cuda.threadIdx.x
#     mat_bit = nb_cuda.threadIdx.y
#
#     vec_cache_blk_shared = nb_cuda.shared.array(shape=64, dtype=nb.complex64)  # 512B
#
#     cache_offset = gate_id_in_blk * mat_n_rows
#
#     pos = compact_arg_sorted[arg_offset + 0]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg_sorted[arg_offset + 1]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit >> 1 & 1
#     index |= val << pos
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit & 1
#     index |= val << pos
#
#     vec_cache_blk_shared[cache_offset + mat_bit] = state_vector[index]
#
#     nb_cuda.syncthreads()
#
#     tmp = nb.complex64(0.0 + 0.0j)
#
#     compact_mat_offset = mat_offset + mat_bit * mat_n_rows
#     tmp += compact_mat[compact_mat_offset] * vec_cache_blk_shared[cache_offset]
#     tmp += compact_mat[compact_mat_offset + 1] * vec_cache_blk_shared[cache_offset + 1]
#     tmp += compact_mat[compact_mat_offset + 2] * vec_cache_blk_shared[cache_offset + 2]
#     tmp += compact_mat[compact_mat_offset + 3] * vec_cache_blk_shared[cache_offset + 3]
#
#     state_vector[index] = tmp
#
#
# @nb_cuda.jit()
# def _gate_on_state_vector_kernel_crz(
#         state_vector,
#         arg_offset,
#         compact_arg,
#         compact_arg_sorted,
#         et,
#         et_neg,
# ):
#     index: int
#     mat_bit: int
#     index = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x
#
#     mat_n_rows = 4
#
#     gate_id_in_blk: int = nb_cuda.threadIdx.x
#     mat_bit = nb_cuda.threadIdx.y
#
#     vec_cache_blk_shared = nb_cuda.shared.array(shape=64, dtype=nb.complex64)  # 512B
#
#     cache_offset = gate_id_in_blk * mat_n_rows
#
#     pos = compact_arg_sorted[arg_offset + 0]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg_sorted[arg_offset + 1]
#     tail = index & ((1 << pos) - 1)
#     index = index >> pos << (pos + 1) | tail
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit >> 1 & 1
#     index |= val << pos
#
#     pos = compact_arg[arg_offset]
#     val = mat_bit & 1
#     index |= val << pos
#
#     vec_cache_blk_shared[cache_offset + mat_bit] = state_vector[index]
#
#     nb_cuda.syncthreads()
#
#     tmp = nb.complex64(0.0 + 0.0j)
#
#     if mat_bit == 0:
#         tmp = vec_cache_blk_shared[cache_offset]
#     elif mat_bit == 1:
#         tmp = vec_cache_blk_shared[cache_offset + 1]
#     elif mat_bit == 2:
#         tmp = vec_cache_blk_shared[cache_offset + 2] * et_neg
#     else:  # mat_bit==3
#         tmp = vec_cache_blk_shared[cache_offset + 3] * et
#
#     state_vector[index] = tmp


# @nb_cuda.jit()
# def _gate_on_state_vector_kernel(
#         state_vector,
#         affect_num,
#         mat_offset,
#         arg_offset,
#         compact_mat,
#         compact_arg,
#         compact_arg_sorted,
# ):
#     index: int
#     mat_bit: int
#     index = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x
#
#     mat_n_rows = 1 << affect_num
#
#     gate_id_in_blk: int = nb_cuda.threadIdx.x
#     mat_bit = nb_cuda.threadIdx.y
#
#     vec_cache_blk_shared = nb_cuda.shared.array(shape=64, dtype=nb.complex64)  # 512B
#
#     cache_offset = gate_id_in_blk * mat_n_rows
#
#     for i in range(affect_num):
#         pos = compact_arg_sorted[arg_offset + i]
#         tail = index & ((1 << pos) - 1)
#         index = index >> pos << (pos + 1) | tail
#     for i in range(affect_num):
#         pos = compact_arg[arg_offset + i]
#         val = mat_bit >> (affect_num - 1 - i) & 1
#         index |= val << pos
#
#     vec_cache_blk_shared[cache_offset + mat_bit] = state_vector[index]
#
#     nb_cuda.syncthreads()
#
#     tmp = nb.complex64(0.0 + 0.0j)
#
#     compact_mat_offset = mat_offset + mat_bit * mat_n_rows
#     for i in range(mat_n_rows):
#         tmp += compact_mat[compact_mat_offset + i] * vec_cache_blk_shared[cache_offset + i]
#
#     state_vector[index] = tmp


# def state_vector_simulation(
#         initial_state: np.ndarray,
#         gates: List[BasicGate],
# ):
#     with nb_cuda.gpus[0]:
#
#         start_time = time()
#
#         mat_offset = 0
#         arg_offset = 0
#         gate_cnt = len(gates)
#         mat_strides = np.empty(shape=gate_cnt, dtype=np.int64)
#         arg_strides = np.empty(shape=gate_cnt, dtype=np.int64)
#
#         for i, gate in enumerate(gates):
#             mat_strides[i] = mat_offset
#             arg_strides[i] = arg_offset
#             arg_offset += len(gate.affectArgs)
#             mat_offset += gate.compute_matrix.shape[0] ** 2
#
#         compact_mat = np.empty(shape=mat_offset, dtype=np.complex64)
#         compact_arg = np.empty(shape=arg_offset, dtype=np.int64)
#         compact_arg_sorted = np.empty(shape=arg_offset, dtype=np.int64)
#
#         for i, gate in enumerate(gates):
#             offset = mat_strides[i]
#             mat_sz = gate.compute_matrix.shape[0] ** 2
#             compact_mat[offset:offset + mat_sz] = gate.compute_matrix.flatten()[:]
#
#             offset = arg_strides[i]
#             affect_num = len(gate.affectArgs)
#             tmp_arr = np.array(gate.affectArgs)
#             compact_arg[offset:offset + affect_num] = tmp_arr[:]
#             compact_arg_sorted[offset:offset + affect_num] = np.sort(tmp_arr)[:]
#
#         end_time = time()
#         data_gather_time = end_time - start_time
#
#         start_time = time()
#
#         d_state_vector = nb_cuda.to_device(initial_state)
#         d_compact_mat = nb_cuda.to_device(compact_mat)
#         d_compact_arg = nb_cuda.to_device(compact_arg)
#         d_compact_arg_sorted = nb_cuda.to_device(compact_arg_sorted)
#
#         end_time = time()
#         data_transfer_time = end_time - start_time
#
#         start_time = time()
#
#         for i, gate in enumerate(gates):
#             affect_num = len(gate.affectArgs)
#             mat_n_rows = gate.compute_matrix.shape[0]
#             gate_in_blk = thread_num_per_block // mat_n_rows
#             thread_per_block = (gate_in_blk, mat_n_rows)
#             blk_cnt = initial_state.shape[0] // thread_num_per_block
#             # if affect_num == 1:
#             #     _gate_on_state_vector_kernel_1bit[blk_cnt, thread_per_block](
#             #         d_state_vector,
#             #         mat_strides[i],
#             #         arg_strides[i],
#             #         d_compact_mat,
#             #         d_compact_arg,
#             #         d_compact_arg_sorted,
#             #     )
#             # elif affect_num == 2:
#             #     if gate.type() == GATE_ID["CRz"]:
#             #         _gate_on_state_vector_kernel_crz[blk_cnt, thread_per_block](
#             #             d_state_vector,
#             #             arg_strides[i],
#             #             d_compact_arg,
#             #             d_compact_arg_sorted,
#             #             gate.compute_matrix[2, 2],
#             #             gate.compute_matrix[3, 3],
#             #         )
#             #     else:
#             #         _gate_on_state_vector_kernel_2bit[blk_cnt, thread_per_block](
#             #             d_state_vector,
#             #             mat_strides[i],
#             #             arg_strides[i],
#             #             d_compact_mat,
#             #             d_compact_arg,
#             #             d_compact_arg_sorted,
#             #         )
#             # else:
#             #     _gate_on_state_vector_kernel[blk_cnt, thread_per_block](
#             #         d_state_vector,
#             #         affect_num,
#             #         mat_strides[i],
#             #         arg_strides[i],
#             #         d_compact_mat,
#             #         d_compact_arg,
#             #         d_compact_arg_sorted,
#             #     )
#
#         end_time = time()
#         cuda_run_time = end_time - start_time
#
#         print()
#         print(f"Data gather time: {data_gather_time:0.4f} s")
#         print(f"Data transfer time: {data_transfer_time:0.4f} s")
#         print(f"CUDA execution time: {cuda_run_time:0.4f} s")
#
#         # return d_state_vector.copy_to_host()

@nb_cuda.jit()
def _gate_on_state_vector_kernel_remake_1bit(
        state_vector,
        mat_offset,
        compact_mat,
        affect_arg,
):
    index_0 = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x

    arg_shift = 1 << affect_arg
    tail = index_0 & (arg_shift - 1)
    index_0 = index_0 >> affect_arg << (affect_arg + 1) | tail
    index_1 = index_0 | arg_shift

    vec_0 = state_vector[index_0]
    vec_1 = state_vector[index_1]

    tmp_0 = nb.complex64(0.0 + 0.0j)
    tmp_1 = nb.complex64(0.0 + 0.0j)

    tmp_0 += compact_mat[mat_offset] * vec_0
    tmp_0 += compact_mat[mat_offset + 1] * vec_1

    tmp_1 += compact_mat[mat_offset + 2] * vec_0
    tmp_1 += compact_mat[mat_offset + 3] * vec_1

    state_vector[index_0] = tmp_0
    state_vector[index_1] = tmp_1


@nb_cuda.jit()
def _gate_on_state_vector_kernel_remake_h(
        state_vector,
        sqrt_2,
        affect_arg,
):
    index_0 = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x

    arg_shift = 1 << affect_arg
    tail = index_0 & (arg_shift - 1)
    index_0 = index_0 >> affect_arg << (affect_arg + 1) | tail
    index_1 = index_0 | arg_shift

    vec_0 = state_vector[index_0]
    vec_1 = state_vector[index_1]

    tmp_0 = sqrt_2 * vec_0 + sqrt_2 * vec_1
    tmp_1 = sqrt_2 * vec_0 - sqrt_2 * vec_1

    state_vector[index_0] = tmp_0
    state_vector[index_1] = tmp_1


@nb_cuda.jit()
def _gate_on_state_vector_kernel_remake_2bit(
        state_vector,
        mat_offset,
        arg_offset,
        compact_mat,
        compact_arg_sorted,
        arg_1_shift,
        arg_2_shift,
        arg_3_shift,
):
    index_0 = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x

    pos = compact_arg_sorted[arg_offset]
    tail = index_0 & ((1 << pos) - 1)
    index_0 = index_0 >> pos << (pos + 1) | tail

    pos = compact_arg_sorted[arg_offset + 1]
    tail = index_0 & ((1 << pos) - 1)
    index_0 = index_0 >> pos << (pos + 1) | tail

    index_1 = index_0 | arg_1_shift
    index_2 = index_0 | arg_2_shift
    index_3 = index_0 | arg_3_shift

    vec_0 = state_vector[index_0]
    vec_1 = state_vector[index_1]
    vec_2 = state_vector[index_2]
    vec_3 = state_vector[index_3]

    tmp_0 = nb.complex64(0)
    tmp_1 = nb.complex64(0)
    tmp_2 = nb.complex64(0)
    tmp_3 = nb.complex64(0)

    tmp_0 += compact_mat[mat_offset + 0] * vec_0
    tmp_0 += compact_mat[mat_offset + 1] * vec_1
    tmp_0 += compact_mat[mat_offset + 2] * vec_2
    tmp_0 += compact_mat[mat_offset + 3] * vec_3

    tmp_1 += compact_mat[mat_offset + 4] * vec_0
    tmp_1 += compact_mat[mat_offset + 5] * vec_1
    tmp_1 += compact_mat[mat_offset + 6] * vec_2
    tmp_1 += compact_mat[mat_offset + 7] * vec_3

    tmp_2 += compact_mat[mat_offset + 8] * vec_0
    tmp_2 += compact_mat[mat_offset + 9] * vec_1
    tmp_2 += compact_mat[mat_offset + 10] * vec_2
    tmp_2 += compact_mat[mat_offset + 11] * vec_3

    tmp_3 += compact_mat[mat_offset + 12] * vec_0
    tmp_3 += compact_mat[mat_offset + 13] * vec_1
    tmp_3 += compact_mat[mat_offset + 14] * vec_2
    tmp_3 += compact_mat[mat_offset + 15] * vec_3

    state_vector[index_0] = tmp_0
    state_vector[index_1] = tmp_1
    state_vector[index_2] = tmp_2
    state_vector[index_3] = tmp_3


@nb_cuda.jit()
def _gate_on_state_vector_kernel_remake_crz(
        state_vector,
        arg_offset,
        compact_arg_sorted,
        arg_2_shift,
        arg_3_shift,
        et,
        et_inv,
):
    index_0 = nb_cuda.blockDim.x * nb_cuda.blockIdx.x + nb_cuda.threadIdx.x

    pos = compact_arg_sorted[arg_offset]
    tail = index_0 & ((1 << pos) - 1)
    index_0 = index_0 >> pos << (pos + 1) | tail

    pos = compact_arg_sorted[arg_offset + 1]
    tail = index_0 & ((1 << pos) - 1)
    index_0 = index_0 >> pos << (pos + 1) | tail

    index_2 = index_0 | arg_2_shift
    index_3 = index_0 | arg_3_shift

    vec_2 = state_vector[index_2]
    vec_3 = state_vector[index_3]

    vec_2 = et * vec_2
    vec_3 = et_inv * vec_3

    state_vector[index_2] = vec_2
    state_vector[index_3] = vec_3


def state_vector_simulation_remake(
        initial_state: np.ndarray,
        gates: List[BasicGate],
        time_measure: bool = False,
) -> np.ndarray:
    with nb_cuda.gpus[0]:

        qubit_num = int(np.log2(initial_state.shape[0]))

        if time_measure:
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
            tmp_arr = qubit_num - 1 - np.array(gate.affectArgs)
            compact_arg[offset:offset + affect_num] = tmp_arr[:]  # reverse bit order
            compact_arg_sorted[offset:offset + affect_num] = np.sort(tmp_arr)[:]

        if time_measure:
            end_time = time()
            data_gather_time = end_time - start_time

            start_time = time()

        d_state_vector = nb_cuda.to_device(initial_state)
        d_compact_mat = nb_cuda.to_device(compact_mat)
        # d_compact_arg = nb_cuda.to_device(compact_arg)
        d_compact_arg_sorted = nb_cuda.to_device(compact_arg_sorted)

        if time_measure:
            end_time = time()
            data_transfer_time = end_time - start_time

            start_time = time()

        for i, gate in enumerate(gates):
            affect_num = len(gate.affectArgs)
            mat_n_rows = gate.compute_matrix.shape[0]
            thread_per_block = thread_num_per_block
            blk_cnt = initial_state.shape[0] // mat_n_rows // thread_num_per_block
            if affect_num == 1:
                if gate.type() == GATE_ID["H"]:
                    _gate_on_state_vector_kernel_remake_h[blk_cnt, thread_per_block](
                        d_state_vector,
                        gate.compute_matrix[0, 0],
                        compact_arg[arg_strides[i]],
                    )
                else:
                    _gate_on_state_vector_kernel_remake_1bit[blk_cnt, thread_per_block](
                        d_state_vector,
                        mat_strides[i],
                        d_compact_mat,
                        compact_arg[arg_strides[i]],
                    )
            elif affect_num == 2:
                arg_1_shift = 1 << compact_arg[arg_strides[i] + 1]
                arg_2_shift = 1 << compact_arg[arg_strides[i]]
                arg_3_shift = arg_2_shift | arg_1_shift
                if gate.type() == GATE_ID["CRz"]:
                    _gate_on_state_vector_kernel_remake_crz[blk_cnt, thread_per_block](
                        d_state_vector,
                        arg_strides[i],
                        d_compact_arg_sorted,
                        arg_2_shift,
                        arg_3_shift,
                        gate.compute_matrix[2, 2],
                        gate.compute_matrix[3, 3],
                    )
                else:
                    _gate_on_state_vector_kernel_remake_2bit[blk_cnt, thread_per_block](
                        d_state_vector,
                        mat_strides[i],
                        arg_strides[i],
                        d_compact_mat,
                        d_compact_arg_sorted,
                        arg_1_shift,
                        arg_2_shift,
                        arg_3_shift,
                    )
            else:
                raise NotImplementedGateException("Only support up to 2 bit gate now.")

        if time_measure:
            end_time = time()
            cuda_run_time = end_time - start_time

            print()
            print(f"Data gather time: {data_gather_time:0.4f} s")
            print(f"Data transfer time: {data_transfer_time:0.4f} s")
            print(f"CUDA execution time: {cuda_run_time:0.4f} s")

        return d_state_vector.copy_to_host()
